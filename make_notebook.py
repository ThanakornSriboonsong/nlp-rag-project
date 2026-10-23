"""Builds RAG_contest.ipynb (the single file submitted to the instructor) from rag.py.

    .venv/Scripts/python make_notebook.py

The notebook contains the full RAG code inline, so it needs no other file besides docs/ and data/question.csv.
"""
import ast
import json

src = open('rag.py', encoding='utf-8').read()
core = src[:src.index("\nif __name__ == '__main__':")].rstrip() + '\n'
core = core.replace('import argparse\n', '')

cells = [
    ('markdown', "# RAG Question Answering - contest submission\n"
     "LLM: `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, no fine-tuning) - retrieval: BM25 + "
     "`BAAI/bge-small-en-v1.5` + `cross-encoder/ms-marco-MiniLM-L-6-v2` - CPU only.\n\n"
     "**How to run:** Runtime -> Change runtime type -> **CPU**, set `PROJECT_DIR` below to the folder that "
     "contains `docs/` (all PDFs) and `data/question.csv`, then **Runtime -> Run all**. The answers are written to "
     "`PROJECT_DIR/output/answer.csv` (`question_id, answer_text, document_id`). Greedy decoding: a re-run gives "
     "the same answers (a re-run after the contest deadline never uses the time-saving fast mode)."),
    ('code', "!pip -q install rank_bm25 sentence-transformers pypdf"),
    ('code', "from google.colab import drive\ndrive.mount('/content/drive')\n\nimport os\n"
             "PROJECT_DIR = '/content/drive/My Drive/Colab Notebooks/NLP_project'   # contains docs/ and data/question.csv\n"
             "print(len([f for f in os.listdir(f'{PROJECT_DIR}/docs') if f.lower().endswith('.pdf')]), 'PDFs in docs/')\n"
             "print(open(f'{PROJECT_DIR}/data/question.csv', encoding='utf-8').read()[:300])"),
    ('code', "# ===================== RAG system (same code as rag.py) =====================\n" + core),
    ('code', "# load PDFs, build BM25 + vector indexes, load the LLM (~5-11 min on Colab CPU)\n"
             "system = RAGSystem(PROJECT_DIR)\nsystem.setup()"),
    ('code', "# data/question.csv -> output/answer.csv  (resumes automatically if the runtime disconnects: Run all again)\n"
             "answers = system.answer_file()\nanswers.head(10)"),
]

ast.parse('\n'.join(s for k, s in cells if k == 'code' and not s.startswith('!pip')))
nb = {'cells': [], 'nbformat': 4, 'nbformat_minor': 0,
      'metadata': {'colab': {'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                   'language_info': {'name': 'python'}}}
for kind, s in cells:
    lines = s.split('\n')
    cell = {'cell_type': kind, 'metadata': {}, 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}
    if kind == 'code':
        cell.update({'execution_count': None, 'outputs': []})
    nb['cells'].append(cell)
json.dump(nb, open('RAG_contest.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote RAG_contest.ipynb')
