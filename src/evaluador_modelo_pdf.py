"""
========================================================
EVALUADOR COMPLETO DE MODELO RAG + EXPORTADOR A PDF
========================================================
Evalúa la respuesta del modelo RAG usando:
- Recuperación: Precision@k, Recall@k, MRR, nDCG
- Generación: Similitud semántica, ROUGE-L, F1 tokens
- Métricas combinadas: QA Accuracy, Top-k retrieval + answer correctness
Cada ejecución agrega una página nueva en el PDF acumulativo.
========================================================
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
from sentence_transformers import SentenceTransformer, util
from rouge import Rouge
from sklearn.metrics import f1_score

# ========================================================
# Modelo de embeddings
# ========================================================
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
rouge = Rouge()

# ========================================================
# Métricas de recuperación
# ========================================================
def precision_recall_at_k(resultados, relevantes, k=3):
    top_k = resultados[:k]
    aciertos = len([doc for doc in top_k if doc in relevantes])
    precision = aciertos / k if k > 0 else 0
    recall = aciertos / len(relevantes) if relevantes else 0
    return precision, recall

def mean_reciprocal_rank(resultados, relevantes):
    for i, doc in enumerate(resultados):
        if doc in relevantes:
            return 1 / (i + 1)
    return 0

def ndcg_at_k(resultados, relevantes, k=5):
    dcg = 0
    for i, doc in enumerate(resultados[:k]):
        if doc in relevantes:
            dcg += 1 / (i + 1)
    idcg = sum(1 / (i + 1) for i in range(min(len(relevantes), k)))
    return dcg / idcg if idcg > 0 else 0

# ========================================================
# Métricas de generación
# ========================================================
def similitud_semantica(generada, esperada):
    gen_emb = model.encode(generada, convert_to_tensor=True)
    exp_emb = model.encode(esperada, convert_to_tensor=True)
    return float(util.cos_sim(gen_emb, exp_emb).item())

def calcular_rouge(respuesta_modelo, respuesta_esperada):
    scores = rouge.get_scores(respuesta_modelo, respuesta_esperada)
    return scores[0]['rouge-l']['f']

def f1_tokens(respuesta_modelo, respuesta_esperada):
    pred_tokens = respuesta_modelo.split()
    ref_tokens = respuesta_esperada.split()
    all_tokens = list(set(pred_tokens + ref_tokens))
    y_true = [1 if t in ref_tokens else 0 for t in all_tokens]
    y_pred = [1 if t in pred_tokens else 0 for t in all_tokens]
    return f1_score(y_true, y_pred)

# ========================================================
# Exportar a PDF
# ========================================================
def guardar_resultados_pdf(
    query, precision, recall, mrr, ndcg, semantica,
    rouge_l, f1, qa_accuracy, topk_answer_correctness,
    score_human=None, filename="evaluacion_rag.pdf"
):
    temp_file = "temp_eval.pdf"
    c = canvas.Canvas(temp_file, pagesize=letter)
    width, height = letter
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 60, "📊 Reporte de Evaluación del Modelo RAG")
    c.setFont("Helvetica", 12)
    c.drawString(80, height - 100, f"🕓 Fecha: {fecha}")
    c.drawString(80, height - 120, f"🔍 Consulta: {query}")
    c.drawString(80, height - 140, f"📈 Precision@k: {precision:.2f}")
    c.drawString(80, height - 160, f"📊 Recall@k: {recall:.2f}")
    c.drawString(80, height - 180, f"🏆 MRR: {mrr:.2f}")
    c.drawString(80, height - 200, f"📌 nDCG@k: {ndcg:.2f}")
    c.drawString(80, height - 220, f"🤖 Similitud semántica: {semantica:.2f}")
    c.drawString(80, height - 240, f"📝 ROUGE-L F1: {rouge_l:.2f}")
    c.drawString(80, height - 260, f"🔹 F1 Tokens: {f1:.2f}")
    c.drawString(80, height - 280, f"🎯 QA Accuracy: {qa_accuracy:.2f}")
    c.drawString(80, height - 300, f"📌 Top-k Answer Correctness: {topk_answer_correctness}")

    if score_human is not None:
        c.drawString(80, height - 320, f"🧑 Human Score: {score_human}")

    c.drawString(80, height - 340, "-" * 60)
    c.showPage()
    c.save()

    if os.path.exists(filename):
        writer = PdfWriter()
        for pdf_file in [filename, temp_file]:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                writer.add_page(page)
        with open(filename, "wb") as f:
            writer.write(f)
        os.remove(temp_file)
    else:
        os.rename(temp_file, filename)

    print(f"✅ Resultados guardados en {filename}")

# ========================================================
# Evaluación completa
# ========================================================
def evaluar_rag_completo_v2(
    query, resultados_recuperados, documentos_relevantes,
    respuesta_modelo, respuesta_esperada, k=5, score_human=None
):
    # Recuperación
    precision, recall = precision_recall_at_k(resultados_recuperados, documentos_relevantes, k)
    mrr = mean_reciprocal_rank(resultados_recuperados, documentos_relevantes)
    ndcg = ndcg_at_k(resultados_recuperados, documentos_relevantes, k)

    # Generación
    semantica = similitud_semantica(respuesta_modelo, respuesta_esperada)
    rouge_l = calcular_rouge(respuesta_modelo, respuesta_esperada)
    f1 = f1_tokens(respuesta_modelo, respuesta_esperada)

    # Métricas combinadas
    qa_accuracy = (recall + f1) / 2
    topk_answer_correctness = int(any(doc in documentos_relevantes for doc in resultados_recuperados[:k]))

    # Guardar PDF
    guardar_resultados_pdf(
        query, precision, recall, mrr, ndcg, semantica,
        rouge_l, f1, qa_accuracy, topk_answer_correctness, score_human
    )

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "similitud": semantica,
        "rouge_l": rouge_l,
        "f1_tokens": f1,
        "qa_accuracy": qa_accuracy,
        "topk_answer_correctness": topk_answer_correctness,
        "human_score": score_human
    }
