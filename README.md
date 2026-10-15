# RAG Question Answering — NLP Project

ระบบตอบคำถามจากเอกสาร PDF 100 ไฟล์ด้วย **RAG** (ค้นเอกสาร → ให้ LLM ตอบจากเอกสาร)
อ่าน `question.csv` → เขียน `answer.csv` (`question_id, answer_text, document_id`) บน **Google Colab CPU**

```
nlp_project/                 ← repo นี้ (โค้ดอย่างเดียว)
├── rag.py                   ← โค้ดทั้งหมด (pipeline, การให้คะแนน, CLI)
├── RAG_colab.ipynb          ← notebook สั้น ๆ สำหรับ Colab: ติดตั้ง → mount Drive → เรียก rag.py
├── requirements.txt
└── README.md

Google Drive: My Drive/Colab Notebooks/NLP_project/   ← ข้อมูล (ไม่อยู่ใน git)
├── rag.py                   ← สำเนาจาก repo (Colab import จากตรงนี้)
├── docs/                    ← PDF 100 ไฟล์
├── data/question_and_answer.csv   ← ชีตเฉลย 357 ข้อ
├── data/question.csv              ← ข้อสอบจริง (วันแข่ง)
└── output/                  ← answer.csv, eval_local.csv, ...
```

---

## 1. กติกาและการให้คะแนน

- CPU เท่านั้น, RAM ≤ 12.7 GB, คำนวณทุกอย่างบนเครื่อง (internet ได้แค่ติดตั้ง/โหลดโมเดล)
- LLM สร้างข้อความได้ **1 ตัว ≤ 2B** → `Qwen/Qwen2.5-1.5B-Instruct` (1.54B)
- วันแข่ง: **100 ข้อ ภายใน 2 ชั่วโมง**

**คะแนนต่อข้อ = 2**

| ส่วน | สูตร |
|---|---|
| `document_id` ถูก | 1 คะแนน |
| yes/no, MCQ | 1 ถ้า `lower(ตอบ) == lower(เฉลย)` |
| ตัวเลข | `1 / (1 + |a − â|)` |
| ข้อความสั้น | `1 / (1 + 2·Levenshtein(a, â) / len(a))` → **ตอบยาวเกินโดนหักหนัก** ตอบว่างยังได้ 0.33 |

ฟังก์ชัน `answer_score()` ใน `rag.py` คิดคะแนนแบบเดียวกับอาจารย์

---

## 2. Pipeline

```
[ครั้งเดียว]  PDF → ล้างข้อความ (NFKC, ขีด, footer) → chunk 150 คำ ซ้อน 30
             → BM25 (คำตรงตัว: ชื่อ รหัส ตัวเลข) + bge-small vectors (ความหมาย)

[ทุกข้อ]     detect_type() → yesno / mcq / number / text
             retrieve(): BM25 top30 + vector top30 → RRF → cross-encoder rerank → 3 chunks
             Qwen:
               yes/no → logit(yes) − logit(no) > YES_MARGIN ?      (ไม่ generate, รูปแบบถูก 100%)
               MCQ    → token A–F ที่คะแนนสูงสุด                   (ไม่ generate)
               number → generate → number_answer() (ดึงตัวเลข / คิด "99*12" ให้)
               text   → generate → clean_text() → shorten_text() → maybe_number()
             document_id = PDF ของ chunk อันดับ 1
```

| โมเดล | หน้าที่ | ขนาด |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | LLM ตัวเดียวที่ generate | 1.54B |
| `BAAI/bge-small-en-v1.5` | embedding | 33M |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker | 22M |

### เทคนิคสำคัญ
- **Hybrid retrieval + RRF** — BM25 เก่งคำเป๊ะ, vector เก่งความหมาย, RRF รวมอันดับ, reranker อ่านคู่ (คำถาม, chunk) เพื่อเรียงใหม่
- **Logit scoring** — yes/no และ MCQ อ่านคะแนน token คำตอบแทนการให้โมเดลพิมพ์ → รูปแบบถูกเสมอ และเร็ว
- **`YES_MARGIN`** — โมเดลชอบตอบ yes เกินจริง จึงต้องให้ yes ชนะ no อย่างน้อยค่านี้
- **`detect_type()`** — แยก MCQ 5 รูปแบบ (`[A]`, `A)`, `A.`, `A.179`, `A TOEIC`), ประโยคบอกเล่า = yes/no, ใช้คำ wh ตัวสุดท้าย (ถูก 353/357)
- **`shorten_text()`** — ตัดคำตอบให้เหลือเฉพาะสิ่งที่ถาม (`Red cap` → `Red`) เพราะสูตร Levenshtein
- **Calculator** — ให้โมเดลเขียน `99*12` แล้ว `safe_eval()` (ผ่าน `ast` อนุญาตแค่ + − × ÷)
- **Deadline guard** — ถ้าคาดว่าจะเกิน `DEADLINE_MIN` (110 นาที) สลับเป็นโหมดเร็ว (2 chunks) ให้ตอบครบทุกข้อ, error → คำตอบสำรอง

---

## 3. ผลลัพธ์ (ชีตเฉลย, ไม่นับ 21 ข้อที่เฉลยผิด)

| | คะแนนคำตอบ (ก่อนปรับ 9d) | document_id อยู่ใน top 3 |
|---|---|---|
| MCQ (89) | 0.876 | 94.4% |
| ตัวเลข (91) | 0.791 | 95.6% |
| ข้อความสั้น (67) | 0.725 | 100% |
| yes/no (89) | 0.742 | 97.8% |
| **รวม** | **0.787** | — |

- `document_id` อันดับ 1: **93.3%** (เพดาน 94.1%) — โหวตหลาย chunk ไม่ช่วย
- ปรับรอบ 9d (prompt yes/no เข้มขึ้น, ตัดคำตอบ text, เลิกตัวอย่างใน prompt ตัวเลข): ข้อที่เคยผิด 81 ข้อ คะแนน 47.1 → 62.0, ข้อที่เคยถูก 37 ข้อไม่เสียเลย
- ความเร็ว Colab CPU ~50 วินาที/ข้อ → 100 ข้อ ≈ 85 นาที + setup ~10 นาที

**ลองแล้วไม่ได้ผล:** int8 quantization (ตอบเพี้ยน, ไม่เร็วขึ้น), llama.cpp wheel สำเร็จรูป (ช้ากว่า), โหวต document_id จาก RRF ทั้งหมด (73%)

---

## 4. วิธีใช้

### บน Colab (ทดสอบจริง / วันแข่ง)
1. คัดลอก `rag.py` ล่าสุดไปไว้ใน `My Drive/Colab Notebooks/NLP_project/`
2. เปิด `RAG_colab.ipynb` ใน Colab → Runtime type **CPU** → **Run all** (กดอนุญาต Drive)
3. วันแข่ง: วาง `question.csv` ใน `data/`, ลบ `output/answer_partial.csv`, ข้ามเซลล์ evaluate

### บนเครื่อง (พัฒนา / ทดลอง — เร็วกว่ามาก)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```
```bash
.venv/Scripts/python rag.py eval --project-dir "G:/My Drive/Colab Notebooks/NLP_project" --n 40
```
```bash
.venv/Scripts/python rag.py summary --project-dir "G:/My Drive/Colab Notebooks/NLP_project"
```
- `--project-dir` ชี้ไปที่โฟลเดอร์ที่มี `docs/` และ `data/` (ผ่าน Google Drive for Desktop)
- ผลบันทึกที่ `output/eval_local.csv` ทีละข้อ รันซ้ำจะทำต่อจากเดิม (ลบไฟล์เพื่อเริ่มใหม่)
- **ความเร็วบนเครื่องไม่ใช่ความเร็วจริง** — จับเวลาบน Colab ก่อนส่งเสมอ

### ปรับค่า
แก้ค่าคงที่ด้านบนของ `rag.py`: `CHUNK_WORDS`, `TOP_K_CONTEXT`, `YES_MARGIN`, `DEADLINE_MIN`, `INSTRUCTIONS`
