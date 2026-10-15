"""RAG question answering over the course PDFs (CPU only, one LLM <= 2B parameters).

Pipeline per question:
    detect type -> hybrid retrieval (BM25 + bge vectors -> RRF -> cross-encoder rerank)
    -> prompt Qwen2.5-1.5B-Instruct with the top chunks -> format the answer
    -> document_id = the PDF of the best chunk

Usage (Colab or local):
    from rag import RAGSystem
    rag = RAGSystem('/content/drive/My Drive/Colab Notebooks/NLP_project')
    rag.setup()                      # load PDFs, build indexes, load the LLM
    rag.answer_file()                # data/question.csv -> output/answer.csv
    rag.evaluate(n=40)               # contest score on data/question_and_answer.csv

Command line:
    python rag.py eval   --project-dir PATH [--n 40]
    python rag.py answer --project-dir PATH
"""
import argparse
import ast
import glob
import operator
import os
import re
import time
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'

CHUNK_WORDS, OVERLAP_WORDS = 150, 30   # chunk size / overlap in words
TOP_K_CANDIDATES = 30                  # candidates taken from BM25 and from vectors
TOP_K_CONTEXT = 3                      # chunks given to the LLM
K_BY_TYPE = {}                         # per-type override, e.g. {'yesno': 4, 'mcq': 4} (no generation -> cheaper)
ALLOW_KNOWLEDGE = False                # let the LLM use its own knowledge when the context lacks the answer
YES_MARGIN = 2.0                       # answer "yes" only if logit(yes) - logit(no) > YES_MARGIN (tuned on 89 sample yes/no: 0.72 -> 0.83)
DEADLINE_MIN = 110                     # contest: 100 questions in 120 min -> switch to fast mode after this

QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
FALLBACK = {'yesno': 'no', 'mcq': 'A', 'number': '0', 'text': ''}

# question_ids whose gold answer in the sample sheet is wrong (marked red) - excluded from scores
BAD_GOLD_IDS = {'22', '38', '85', '132', '156', '170', '178', '197', '214', '225', '242', '259', '260',
                '286', '303', '320', '330', '335', '337', '347', '351'}


# ----------------------------------------------------------------------------- documents
def normalize(t):
    t = unicodedata.normalize('NFKC', t)                     # ligatures -> fi, superscripts -> digits
    t = re.sub(r'[‐-―−]', '-', t)              # fancy dashes -> '-'
    t = t.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    t = re.sub(r'^.*Fictional document created for testing.*$', '', t, flags=re.M)  # repeated footer
    return t


def load_documents(docs_dir):
    """one entry per PDF: {'name', 'text'} (pages joined in order)"""
    from pypdf import PdfReader
    docs = []
    for path in sorted(glob.glob(os.path.join(docs_dir, '*.pdf'))):
        pages = [p.extract_text() or '' for p in PdfReader(path).pages]
        docs.append({'name': os.path.basename(path), 'text': normalize('\n'.join(pages))})
    return docs


def chunk_doc(doc):
    lines = [l.strip() for l in doc['text'].split('\n') if l.strip()]
    file_title = re.sub(r'^(Copy of )?\d+_|\.pdf$', '', doc['name']).replace('_', ' ')
    title = lines[0][:120] if lines else ''
    words = ' '.join(lines).split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [{'doc': doc['name'], 'text': f'[{file_title} | {title}] ' + ' '.join(words[i:i + CHUNK_WORDS])}
            for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step)]


def doc_id_of(name):
    """'Copy of 057_peron...pdf' -> 57"""
    m = re.search(r'(\d+)_', name)
    return int(m.group(1)) if m else -1


STOP = set('a an the of to in on at for by with and or is are was were be been it this that '
           'as from which who whom what when where why how did does do according'.split())


def tokenize(t):
    """BM25 tokens; keeps codes and numbers whole: "pp-02", "9.4", "11,800" """
    toks = re.findall(r'[a-z0-9]+(?:[-.,/][a-z0-9]+)*', t.lower())
    return [x for x in toks if x not in STOP]


# ----------------------------------------------------------------------------- question parsing
BRACKET_RE = re.compile(r'\[([A-F])\]\s*')                          # [A] ...
PUNCT_RE = re.compile(r'(?:^|(?<=\s))\(?([A-F])[\.\)]\s*(?=\S)')     # A) ...  A. ...  A.179
LINE_RE = re.compile(r'(?:^|(?<=\n))[ \t]*([A-F])[ \t]+(?=\S)')      # A TOEIC 100 (on its own line)


def _seq(matches):
    seq = []
    for m in matches:                                               # keep only A, B, C... in order
        if len(seq) < 6 and m.group(1) == 'ABCDEF'[len(seq)]:
            seq.append(m)
    return seq


def split_mcq(q):
    """-> (stem, [(letter, option_text), ...]); options list is empty for non-MCQ questions"""
    for rx in (BRACKET_RE, PUNCT_RE, LINE_RE):
        seq = _seq(rx.finditer(q))
        if len(seq) >= 2:
            break
    else:
        return ' '.join(q.split()), []
    opts = []
    for i, m in enumerate(seq):
        end = seq[i + 1].start() if i + 1 < len(seq) else len(q)
        text = re.sub(r'\(correct answer\)', '', q[m.end():end], flags=re.I)
        opts.append((m.group(1), ' '.join(text.split())))
    return ' '.join(q[:seq[0].start()].split()), opts


AUX = r'(?:is|are|was|were|do|does|did|can|could|will|would|has|have|had|should|shall|may|might|must)'
YESNO_RE = re.compile(rf'(?:^|[,:.;]\s+|\bis it true that\s+){AUX}\b', re.I)
WH_RE = re.compile(r"\b(?:what|what's|which|who|whom|whose|where|when|why|how)\b", re.I)
NUM_RE = re.compile(r"\bhow (?:many|much|long|far|old)\b|\bwhat (?:year|percentage|percent|value)\b"
                    r"|\bnumber\b|\bprice\b|\bfee\b|\blevy\b|\bcharge\b|\bcost\b|\bsurcharge\b|\ballowance\b"
                    r"|\bdeposit\b|\bminimum age\b|\bspeed\b|\bdiameter\b|\bdistance\b|\bvolume\b|\bpercentage\b"
                    r"|\(in [^)]*\)|\bin (?:working |calendar )?(?:baht|thb|clp|sgd|eur|usd|km|kw|kg|m|mm|metres"
                    r"|meters|millimetres|minutes|hours|days)\b|%|\btotal\b|\bmileage\b|\bdigit\b", re.I)
PHONE_RE = re.compile(r'\b(?:phone|serial|contact|registration|account) number\b', re.I)


def detect_type(q):
    """'mcq' | 'yesno' | 'number' | 'text'  (353/357 correct on the sample sheet)"""
    stem, opts = split_mcq(q)
    if opts:
        return 'mcq'
    yn, wh = YESNO_RE.search(stem), WH_RE.search(stem)
    if not wh or (yn and yn.start() < wh.start()):
        return 'yesno'                                  # "Is/Does/Can ...?" or a statement to verify
    last = WH_RE.findall(stem)[-1].lower()              # the real question word is usually the last one
    if last in ('who', 'whom', 'whose', 'why') or PHONE_RE.search(stem):
        return 'text'
    if last == 'which':
        return 'number' if re.search(r'\bwhich (?:\w+ ){0,2}(?:number|year)\b', stem, re.I) else 'text'
    if NUM_RE.search(stem):
        return 'number'
    return 'text'


# ----------------------------------------------------------------------------- answer formatting
WORD_NUM = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'.split())}
SCALE = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9}
MONTHS = 'january february march april may june july august september october november december'.split()


def to_number(text):
    t = text.replace(',', '')
    m = re.search(r'-?\d+(?:\.\d+)?', t)
    if m:
        v = float(m.group())
        s = re.match(r'\s*(thousand|million|billion)\b', t[m.end():], re.I)
        if s:
            v *= SCALE[s.group(1).lower()]                 # "11.8 million" -> 11800000
    else:
        w = [WORD_NUM[x] for x in re.findall(r'[a-z]+', t.lower()) if x in WORD_NUM]
        if not w:
            return '0'
        v = float(w[0])
    v = round(v, 6)
    return str(int(v)) if v.is_integer() else str(v)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def safe_eval(expr):
    """evaluate "99*12" safely: numbers, + - * / and brackets only"""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(expr)
    return ev(ast.parse(expr, mode='eval').body)


def number_answer(raw):
    """model output -> digits; computes "99*12" style expressions"""
    s = raw.replace(',', '').replace('×', '*').replace('÷', '/')
    m = re.search(r'[\d.\s+\-*/()]*\d\s*[+\-*/]\s*\(?\s*\d[\d.\s+\-*/()]*', s)
    if m and not re.search(r'million|billion|thousand', s, re.I):
        try:
            return to_number(str(round(safe_eval(m.group().strip()), 4)))
        except Exception:
            pass
    return to_number(raw)


def maybe_number(ans):
    """'600 baht' / 'CLP 4600' / '0.67' -> number; keeps 'Room 204', '02 118 4630', 'SK-05', '22:00', dates"""
    if ':' in ans or any(m in ans.lower() for m in MONTHS):
        return ans
    if len(re.findall(r'\d[\d,]*(?:\.\d+)?', ans)) != 1 or len(ans.split()) > 3:
        return ans
    m = re.match(r'^(?:[A-Za-z]{3}\s*)?(-?\d[\d,]*(?:\.\d+)?)(?![\w-])', ans.strip())
    return to_number(ans.strip()[m.start(1):]) if m else ans


def clean_text(a):
    a = a.strip().split('\n')[0].strip()
    a = re.sub(r'^(the answer is|answer)\s*[:\-]?\s*', '', a, flags=re.I)
    a = a.strip(' "\'*')
    # drop a trailing period on short answers, but keep "Co., Ltd." and full-sentence answers
    if a.endswith('.') and len(a.split()) <= 6 and not re.search(r'\b(?:ltd|co|inc|jr|sr)\.$', a, re.I):
        a = a[:-1]
    return a.strip(' "\'*')


HONORIFIC_RE = re.compile(r'^(?:mr|mrs|ms|miss|dr|khun|prof)\.?\s+', re.I)


def shorten_text(ans, question):
    """short-text scoring is 1/(1 + 2*Levenshtein/len(gold)): extra words cost a lot, so trim to what was asked"""
    a = ans.strip()
    if not a:
        return a
    qwords = {w.lower() for w in re.findall(r"[A-Za-z][\w'-]+", question) if len(w) >= 4}
    words = a.split()
    if len(words) >= 5:
        quoted = re.findall(r'"([^"]{1,40})"', a)
        if len(quoted) == 1:                             # Mark "ABS" next to ...  -> ABS
            return quoted[0].strip()
        for i in range(1, len(words)):                   # "X decides on ..." -> "X" (cut at the question's verb)
            w = re.sub(r'\W', '', words[i]).lower()
            if len(w) >= 4 and w in qwords and words[i][0].islower():
                a = ' '.join(words[:i]).rstrip(' ,;:')
                a = re.sub(r'^the\s+', '', a, flags=re.I)
                break
    a = re.split(r'\s+\(|\s+on\s+\d|\s+at\s+\d', a)[0].strip(' ,;:')   # drop "(06:00 to 24:00)", "on 053 ..."
    m = re.search(r'\s+(?:at|in|of|for)\s+(.+)$', a)
    if m and m.group(1).lower() in question.lower():
        a = a[:m.start()]                                # "supervisor at Gat Sv. Petra" when the question says it
    a = HONORIFIC_RE.sub('', a)
    if re.match(r'^\s*who\b', question, re.I) and ',' in a:
        a = a.split(',')[0].strip()                      # "Wichai Lertpanya, Plant Manager" -> name only
    words = a.split()
    if len(words) >= 2:                                  # "Red cap" for "what color cap" -> "Red"
        last = re.sub(r'\W', '', words[-1])
        qall = {w.lower() for w in re.findall(r'[A-Za-z]+', question) if len(w) >= 3}
        if len(last) >= 3 and last.islower() and any(q.startswith(last.lower()[:5]) for q in qall):
            a = ' '.join(words[:-1])
    return a.strip(' ,;:')


# ----------------------------------------------------------------------------- contest scoring
def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def gold_type(gold):
    g = str(gold).strip()
    if g.lower() in ('yes', 'no'):
        return 'yesno'
    if re.fullmatch(r'[A-F]', g):
        return 'mcq'
    if re.fullmatch(r'-?\d+(\.\d+)?', g):
        return 'number'
    return 'text'


def answer_score(pred, gold):
    """contest answer_text score (0..1); the question type comes from the gold answer"""
    pred, gold = str(pred).strip(), str(gold).strip()
    t = gold_type(gold)
    if t in ('yesno', 'mcq'):
        return float(pred.lower() == gold.lower())
    if t == 'number':
        try:
            return 1 / (1 + abs(float(gold) - float(pred)))
        except ValueError:
            return 0.0
    return 1 / (1 + 2 * levenshtein(gold, pred) / len(gold)) if gold else 0.0


# ----------------------------------------------------------------------------- prompts
SYSTEM = 'You are a precise assistant. Answer using only the information in the context.'
SYSTEM_KNOWLEDGE = ('You are a precise assistant. Answer using the information in the context. '
                    'If the context does not contain the answer (for example a general NLP or machine '
                    'learning concept), answer from your own knowledge.')
INSTRUCTIONS = {
    'yesno': 'Answer with only yes or no. Answer yes only if the context clearly confirms every detail '
             '(names, numbers, days, amounts, roles). If any detail is different or not stated, answer no.',
    'mcq': 'Answer with only the letter of the correct option.',
    'number': 'Answer with only a number. If the answer needs a calculation, write only the arithmetic '
              'expression using numbers from the context (for example 12*30 or 450-200). No units and no words.',
    'text': 'Answer with only the shortest exact words from the context that answer the question - '
            'a name, place, time, code or short phrase - not a sentence. Do not repeat the question. '
            'Example: "Who approves leave?" -> Somchai Dee (not "Mr. Somchai Dee, HR Manager, approves leave.")',
}
YES_NO = {'yes': ['yes', 'Yes', ' yes', ' Yes'], 'no': ['no', 'No', ' no', ' No']}


def save_csv(df, path):
    """write to a temp file, then rename: a crash or power loss never leaves a half-written file"""
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_progress(path):
    """rows saved by an earlier (interrupted) run; [] if missing or unreadable"""
    if not os.path.exists(path):
        return []
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict('records')
    except Exception as e:
        os.replace(path, path + '.broken')
        print(f'could not read {path} ({e!r}); moved it to .broken and starting over')
        return []


# ----------------------------------------------------------------------------- the system
class RAGSystem:
    def __init__(self, project_dir='.', threads=None):
        self.project_dir = project_dir
        self.docs_dir = os.path.join(project_dir, 'docs')
        self.data_dir = os.path.join(project_dir, 'data')
        self.output_dir = os.path.join(project_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.threads = threads or os.cpu_count()
        self.run_start = time.time()                     # for the contest deadline guard

    # ---- setup
    def setup(self, load_llm=True):
        import torch
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder
        torch.set_num_threads(self.threads)
        t = time.time()
        self.documents = load_documents(self.docs_dir)
        self.chunks = [c for d in self.documents for c in chunk_doc(d)]
        print(f'{len(self.documents)} documents, {len(self.chunks)} chunks')
        self.bm25 = BM25Okapi([tokenize(c['text']) for c in self.chunks])
        self.embedder = SentenceTransformer(EMBED_MODEL, device='cpu')
        self.chunk_emb = self.embedder.encode([c['text'] for c in self.chunks], batch_size=32,
                                              normalize_embeddings=True, show_progress_bar=True)
        self.reranker = CrossEncoder(RERANK_MODEL, device='cpu')
        print(f'indexes ready ({time.time() - t:.0f}s)')
        if load_llm:
            self.load_llm()

    def load_llm(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        t = time.time()
        self.tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=torch.float32,
                                                        low_cpu_mem_usage=True).eval()
        print(f'LLM loaded ({time.time() - t:.0f}s)')

    # ---- retrieval
    def retrieve(self, query, keyword_query=None, k=TOP_K_CONTEXT):
        bm = self.bm25.get_scores(tokenize(keyword_query or query))
        qv = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        dense = self.chunk_emb @ qv
        rrf = {}                                         # Reciprocal Rank Fusion
        for scores in (bm, dense):
            for rank, idx in enumerate(np.argsort(-scores)[:TOP_K_CANDIDATES]):
                rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
        cand = sorted(rrf, key=rrf.get, reverse=True)[:TOP_K_CANDIDATES]
        rr = self.reranker.predict([(query, self.chunks[i]['text']) for i in cand])
        return [self.chunks[cand[i]] for i in np.argsort(-rr)[:k]]

    # ---- LLM calls
    def build_prompt(self, context, question, qtype):
        msgs = [{'role': 'system', 'content': SYSTEM_KNOWLEDGE if ALLOW_KNOWLEDGE else SYSTEM},
                {'role': 'user', 'content': f'Context:\n{context}\n\nQuestion: {question}\n\n{INSTRUCTIONS[qtype]}'}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _last_logits(self, prompt):
        import torch
        with torch.inference_mode():
            inputs = self.tok(prompt, return_tensors='pt')
            try:
                return self.llm(**inputs, logits_to_keep=1).logits[0, -1]   # skip the vocab matmul elsewhere
            except TypeError:                                                # older transformers
                return self.llm(**inputs).logits[0, -1]

    def _label_score(self, logits, variants):
        return max(logits[self.tok.encode(v, add_special_tokens=False)[0]].item() for v in variants)

    def choose(self, prompt, label_variants):
        """logit scoring: the answer is the label whose first token scores highest (format always valid)"""
        logits = self._last_logits(prompt)
        scores = {lab: self._label_score(logits, vs) for lab, vs in label_variants.items()}
        return max(scores, key=scores.get)

    def yes_no_diff(self, prompt):
        logits = self._last_logits(prompt)
        return self._label_score(logits, YES_NO['yes']) - self._label_score(logits, YES_NO['no'])

    def generate(self, prompt, max_new_tokens=24):
        import torch
        with torch.inference_mode():
            inputs = self.tok(prompt, return_tensors='pt')
            out = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # ---- one question
    def answer_question(self, q, qtype=None, fast=False):
        """-> (type, answer, retrieved_chunks, yes_no_diff); fast=True: 2 chunks + 16 new tokens"""
        new_tokens = 16 if fast else 24
        qtype = qtype or detect_type(q)
        k = 2 if fast else K_BY_TYPE.get(qtype, TOP_K_CONTEXT)
        stem, opts = split_mcq(q)
        ctx = lambda cs: '\n\n'.join(f'[{i + 1}] {c["text"]}' for i, c in enumerate(cs))
        if qtype == 'mcq' and opts:
            used = self.retrieve(stem, stem + ' ' + ' '.join(t for _, t in opts), k=k)
            qtext = stem + '\n' + '\n'.join(f'{l}. {t}' for l, t in opts)
            ans = self.choose(self.build_prompt(ctx(used), qtext, 'mcq'), {l: [l, ' ' + l] for l, _ in opts})
            return qtype, ans, used, None
        if qtype == 'mcq':
            qtype = 'text'
        used = self.retrieve(stem, k=k)
        p = self.build_prompt(ctx(used), stem, qtype)
        diff = None
        if qtype == 'yesno':
            diff = self.yes_no_diff(p)
            ans = 'yes' if diff > YES_MARGIN else 'no'
        elif qtype == 'number':
            ans = number_answer(self.generate(p, new_tokens))
        else:
            ans = maybe_number(shorten_text(clean_text(self.generate(p, new_tokens)), stem))
        return qtype, ans, used, diff

    @staticmethod
    def pick_doc_id(used):
        """document_id = PDF of the best-ranked chunk (93% on the sample sheet, ceiling 94%)"""
        return doc_id_of(used[0]['doc']) if used else -1

    # ---- contest run: question.csv -> answer.csv
    def answer_file(self, question_csv=None, answer_csv=None):
        question_csv = question_csv or os.path.join(self.data_dir, 'question.csv')
        answer_csv = answer_csv or os.path.join(self.output_dir, 'answer.csv')
        partial_csv = answer_csv.replace('.csv', '_partial.csv')
        qdf = pd.read_csv(question_csv, dtype=str, keep_default_na=False)
        q_col = 'question_text' if 'question_text' in qdf.columns else \
            next(c for c in qdf.columns if 'question' in c.lower() and 'id' not in c.lower())
        id_col = 'question_id' if 'question_id' in qdf.columns else \
            next((c for c in qdf.columns if c.lower() in ('id', 'qid', 'no')), None)

        answers, docs = {}, {}
        for r in load_progress(partial_csv):              # resume after a disconnect
            answers[int(r['idx'])], docs[int(r['idx'])] = r['answer'], r['doc']

        todo = [i for i in range(len(qdf)) if i not in answers]
        print(f'{len(qdf)} questions, {len(todo)} to answer')
        t0, fast = time.time(), False
        for j, i in enumerate(todo):
            q = qdf[q_col][i]
            if not fast and j >= 1:                       # deadline guard
                projected = (time.time() - self.run_start) + (len(todo) - j) * (time.time() - t0) / j
                if projected > DEADLINE_MIN * 60:
                    fast = True
                    print(f'!!! projected {projected / 60:.0f} min > {DEADLINE_MIN} -> FAST MODE '
                          f'for the last {len(todo) - j} questions')
            try:
                qt, a, used, _ = self.answer_question(q, fast=fast)
                doc = self.pick_doc_id(used)
            except Exception as e:                        # never leave a question unanswered
                qt = detect_type(q)
                a = FALLBACK[qt]
                try:
                    doc = self.pick_doc_id(self.retrieve(split_mcq(q)[0]))
                except Exception:
                    doc = -1
                print(f'{i}: ERROR {e!r} -> fallback')
            answers[i], docs[i] = a, doc
            save_csv(pd.DataFrame({'idx': list(answers), 'answer': list(answers.values()),
                                   'doc': [docs[k] for k in answers]}), partial_csv)
            print(f'{i}: [{qt}] {a!r} doc={doc}  ({(time.time() - self.run_start) / 60:.1f} min'
                  f'{", FAST" if fast else ""})')

        out = pd.DataFrame()
        if id_col:
            out['question_id'] = qdf[id_col]
        out['answer_text'] = [answers[i] for i in range(len(qdf))]
        out['document_id'] = [docs[i] for i in range(len(qdf))]
        save_csv(out, answer_csv)
        print('saved', answer_csv)
        return out

    # ---- evaluation with the contest formula
    def evaluate(self, gold_csv=None, n=None, ids=None, out_csv=None, seed=0, types=None):
        """answers the sample questions and scores them like the grader (2 points per question).
        Saves after every question and resumes from out_csv. types: e.g. ['yesno', 'mcq'] (by gold answer)"""
        gold_csv = gold_csv or os.path.join(self.data_dir, 'question_and_answer.csv')
        out_csv = out_csv or os.path.join(self.output_dir, 'eval_local.csv')
        gold = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
        gold = gold[~gold['question_id'].isin(BAD_GOLD_IDS)]
        if types:
            gold = gold[gold['answer_text'].map(gold_type).isin(types)]
        if ids is not None:
            gold = gold[gold['question_id'].isin({str(i) for i in ids})]
        elif n:
            gold = gold.sample(min(n, len(gold)), random_state=seed)

        rows = load_progress(out_csv)
        done = {r['id'] for r in rows}
        t0, k = time.time(), 0
        for _, r in gold.iterrows():
            if r['question_id'] in done:
                continue
            t = time.time()
            qt, pred, used, diff = self.answer_question(r['question_text'])
            doc = self.pick_doc_id(used)
            try:
                gdoc = int(float(r['document_id']))
            except ValueError:
                gdoc = -1
            rows.append({'id': r['question_id'], 'type': qt, 'gtype': gold_type(r['answer_text']),
                         'pred': pred, 'gold': r['answer_text'],
                         'ans_score': answer_score(pred, r['answer_text']),
                         'doc': doc, 'gold_doc': gdoc, 'doc_score': float(doc == gdoc),
                         'yn_diff': '' if diff is None else round(diff, 3), 'sec': round(time.time() - t, 1),
                         'question': r['question_text'][:150]})
            save_csv(pd.DataFrame(rows), out_csv)
            k += 1
            print(f"{len(rows)}/{len(gold)} [{qt}] pred={pred!r} gold={r['answer_text']!r} "
                  f"score={rows[-1]['ans_score']:.2f} doc={'ok' if doc == gdoc else 'X'}  "
                  f"({(time.time() - t0) / k:.1f}s/q)")
        return summarize(out_csv)


def summarize(eval_csv):
    res = pd.read_csv(eval_csv, keep_default_na=False)
    res = res[~res['id'].astype(str).isin(BAD_GOLD_IDS)]
    table = res.groupby('gtype').agg(n=('ans_score', 'size'), answer=('ans_score', 'mean'),
                                     document_id=('doc_score', 'mean'), sec=('sec', 'mean')).round(3)
    print(table)
    total = res['ans_score'].mean() + res['doc_score'].mean()
    print(f"\nanswer {res['ans_score'].mean():.3f} + document_id {res['doc_score'].mean():.3f} "
          f"= {total:.3f} / 2 per question  (n={len(res)}, {res['sec'].mean():.1f}s/question)")
    return res


def compare(base_csv, new_csv):
    """score difference between two eval runs on the questions both of them answered"""
    a = pd.read_csv(base_csv, dtype={'id': str}, keep_default_na=False).set_index('id')
    b = pd.read_csv(new_csv, dtype={'id': str}, keep_default_na=False).set_index('id')
    ids = a.index.intersection(b.index).difference(list(BAD_GOLD_IDS))
    a, b = a.loc[ids], b.loc[ids]
    a['pts'] = a['ans_score'] + a['doc_score']
    b['pts'] = b['ans_score'] + b['doc_score']
    t = pd.DataFrame({'n': a.groupby('gtype').size(),
                      'base': a.groupby('gtype')['pts'].mean(), 'new': b.groupby('gtype')['pts'].mean(),
                      'base_sec': a.groupby('gtype')['sec'].mean(), 'new_sec': b.groupby('gtype')['sec'].mean()})
    print(t.round(3))
    d = b['pts'] - a['pts']
    print(f"\n{len(ids)} questions: {a['pts'].mean():.3f} -> {b['pts'].mean():.3f} points/question  "
          f"(better {(d > 0.01).sum()}, worse {(d < -0.01).sum()}; "
          f"{a['sec'].mean():.1f}s -> {b['sec'].mean():.1f}s per question)")
    for i in d[d.abs() > 0.01].sort_values().index:
        print(f"  {i:>4} [{a.loc[i, 'gtype']}] {a.loc[i, 'pred']!r} -> {b.loc[i, 'pred']!r}  gold={a.loc[i, 'gold']!r}"
              f"  doc {a.loc[i, 'doc']}->{b.loc[i, 'doc']} (gold {a.loc[i, 'gold_doc']})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['eval', 'answer', 'summary', 'compare'])
    ap.add_argument('--project-dir', default=os.environ.get('RAG_PROJECT_DIR', '.'))
    ap.add_argument('--n', type=int, default=None, help='eval: random sample size (default: all)')
    ap.add_argument('--types', default=None, help='eval: only these gold types, e.g. yesno,mcq')
    ap.add_argument('--threads', type=int, default=None, help='CPU threads (default: all cores)')
    ap.add_argument('--out', default=None, help='eval: results csv (resumes if it exists)')
    ap.add_argument('--base', default=None, help='compare: baseline results csv (compared with --out)')
    # experiment switches (defaults = the settings above)
    ap.add_argument('--rerank', default=None, help='cross-encoder model name')
    ap.add_argument('--k-yn-mcq', type=int, default=None, help='chunks for yes/no + MCQ questions')
    ap.add_argument('--knowledge', action='store_true', help='allow the LLM to use its own knowledge')
    args = ap.parse_args()
    if args.rerank:
        RERANK_MODEL = args.rerank
    if args.k_yn_mcq:
        K_BY_TYPE.update(yesno=args.k_yn_mcq, mcq=args.k_yn_mcq)
    if args.knowledge:
        ALLOW_KNOWLEDGE = True
    if args.command == 'summary':
        summarize(args.out or os.path.join(args.project_dir, 'output', 'eval_local.csv'))
    elif args.command == 'compare':
        compare(args.base, args.out)
    else:
        rag = RAGSystem(args.project_dir, threads=args.threads)
        rag.setup()
        if args.command == 'eval':
            rag.evaluate(n=args.n, out_csv=args.out, types=args.types.split(',') if args.types else None)
        else:
            rag.answer_file()
