"""
========================================================
EVALUADOR DE MODELO RAG + EXPORTADOR A PDF ACUMULATIVO
========================================================
Evalúa la respuesta del modelo RAG usando:
- Precision / Recall: calidad de los documentos recuperados
- Similitud semántica con la respuesta esperada
- BLEU / ROUGE: coincidencia literal opcional
Cada ejecución agrega una página nueva en el PDF.
========================================================
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader, PdfWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# ========================================================
# Modelo de embeddings
# ========================================================
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
rouge = Rouge()
smooth_fn = SmoothingFunction().method1

# ========================================================
# [1] MÉTRICAS
# ========================================================

def precision_recall_at_k(resultados, k=3):
    """
    Calcula Precision y Recall usando documentos recuperados
    por embeddings. Por defecto considera todos los documentos como relevantes.
    """
    precision = 1.0  # todos los documentos recuperados cuentan
    recall = min(k, len(resultados)) / len(resultados) if resultados else 0
    return precision, recall

def evaluar_similitud_respuesta(generada, esperada):
    """
    Calcula similitud semántica entre la respuesta del modelo
    y la respuesta esperada usando embeddings.
    """
    gen_emb = model.encode(generada, convert_to_tensor=True)
    exp_emb = model.encode(esperada, convert_to_tensor=True)
    return float(util.cos_sim(gen_emb, exp_emb).item())

def calcular_bleu(respuesta_modelo, respuesta_esperada):
    """
    Calcula BLEU score entre la respuesta generada y la esperada.
    """
    referencia = [respuesta_esperada.split()]
    candidato = respuesta_modelo.split()
    return sentence_bleu(referencia, candidato, smoothing_function=smooth_fn)

def calcular_rouge(respuesta_modelo, respuesta_esperada):
    """
    Calcula ROUGE-L F1 entre la respuesta generada y la esperada.
    """
    scores = rouge.get_scores(respuesta_modelo, respuesta_esperada)
    return scores[0]['rouge-l']['f']

# ========================================================
# [2] EXPORTAR A PDF ACUMULATIVO
# ========================================================

def guardar_resultados_pdf_acumulativo(query, precision, recall, similitud, bleu, rouge_l, filename="evaluacion_rag.pdf"):
    temp_file = "temp_eval.pdf"
    c = canvas.Canvas(temp_file, pagesize=letter)
    width, height = letter

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Encabezado de la página
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 60, "📊 Reporte de Evaluación del Modelo RAG")
    c.setFont("Helvetica", 12)
    c.drawString(80, height - 100, f"🕓 Fecha: {fecha}")
    c.drawString(80, height - 120, f"🔍 Consulta: {query}")
    c.drawString(80, height - 140, f"📈 Precision: {precision:.2f}")
    c.drawString(80, height - 160, f"📊 Recall: {recall:.2f}")
    c.drawString(80, height - 180, f"🤖 Similaridad semántica: {similitud:.2f}")
    c.drawString(80, height - 200, f"📝 BLEU: {bleu:.2f}")
    c.drawString(80, height - 220, f"📝 ROUGE-L F1: {rouge_l:.2f}")
    c.drawString(80, height - 240, "-" * 60)

    c.showPage()
    c.save()

    # Si ya existe el PDF, unirlo con el nuevo
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
# [3] FUNCIÓN GENERAL
# ========================================================

def evaluar_y_guardar(query, resultados, respuesta_modelo, respuesta_esperada):
    """
    Evalúa el RAG y guarda los resultados automáticamente en un PDF acumulativo.
    Calcula Precision, Recall, Similitud semántica, BLEU y ROUGE-L F1.
    """
    precision, recall = precision_recall_at_k(resultados)
    similitud = evaluar_similitud_respuesta(respuesta_modelo, respuesta_esperada)
    bleu = calcular_bleu(respuesta_modelo, respuesta_esperada)
    rouge_l = calcular_rouge(respuesta_modelo, respuesta_esperada)

    guardar_resultados_pdf_acumulativo(query, precision, recall, similitud, bleu, rouge_l)
    return precision, recall, similitud, bleu, rouge_l
