# ==== Webapp OCR -> Email (.EML) ====
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Tesseract + idiomas + libs necesarias
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libmagic1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== requirements =====
RUN cat <<'REQ' > requirements.txt
Flask==3.0.3
pillow==10.4.0
pytesseract==0.3.13
opencv-python-headless==4.10.0.84
python-magic==0.4.27
gunicorn==22.0.0
REQ

RUN pip install --no-cache-dir -r requirements.txt

# ===== app.py =====
RUN cat <<'PY' > app.py
from flask import Flask, request, redirect, url_for, send_file, render_template_string
from io import BytesIO
from PIL import Image
import pytesseract, cv2, numpy as np, re, os, base64, traceback

try:
    from msgbuilder import Message as MsgMessage  # opcional
    HAS_MSGBUILDER = True
except Exception:
    HAS_MSGBUILDER = False

app = Flask(__name__)

INDEX_HTML = """\
<!doctype html>
<html lang="es"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OCR a Email</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Helvetica Neue',Arial;margin:24px}
.card{max-width:920px;margin:0 auto;padding:20px;border:1px solid #ddd;border-radius:16px;box-shadow:0 5px 14px rgba(0,0,0,.06)}
.row{display:flex;gap:16px;align-items:flex-start}.col{flex:1}.preview{border:1px dashed #ccc;padding:8px;border-radius:12px;min-height:160px;display:flex;align-items:center;justify-content:center;background:#fafafa}
.kpi{display:inline-block;padding:8px 12px;border-radius:999px;background:#f0f0ff;margin-right:8px}button{padding:10px 16px;border-radius:10px;border:1px solid #222;background:#111;color:#fff;cursor:pointer}
button.secondary{background:#fff;color:#111}.result{background:#f7fff2;border:1px solid #bfe0b2;padding:12px;border-radius:12px}.err{background:#fff4f4;border:1px solid #f3b1b1;padding:12px;border-radius:12px;color:#7a0000}
</style></head>
<body><div class="card">
<h1>Subir imagen → Extraer producto y cantidad → Generar email</h1>
<p>Sube una captura/nota. Detecto <span class="kpi">Producto</span> y <span class="kpi">Cantidad</span> por OCR y te doy un <b>.EML</b>.</p>
<form action="{{ url_for('process') }}" method="post" enctype="multipart/form-data">
  <div class="row"><div class="col">
    <label>Imagen:</label><input type="file" name="image" accept="image/*" required />
    <label>Destinatario (To):</label><input type="email" name="to" placeholder="ventas@tuempresa.com" style="width:100%;padding:8px;border-radius:8px;border:1px solid #ccc" required/>
    <label>Remitente (From):</label><input type="email" name="sender" placeholder="noreply@tuempresa.com" style="width:100%;padding:8px;border-radius:8px;border:1px solid #ccc" required/>
    <label>Asunto:</label><input type="text" name="subject" placeholder="Pedido detectado" style="width:100%;padding:8px;border-radius:8px;border:1px solid #ccc"/>
    <label>Mensaje adicional (opcional):</label><textarea name="note" rows="3" style="width:100%;padding:8px;border-radius:8px;border:1px solid #ccc"></textarea>
    <div style="margin-top:10px"><button type="submit">Procesar y generar email</button></div>
  </div><div class="col"><div class="preview">La vista previa aparecerá aquí después de subir.</div></div></div>
</form>

{% if error %}
  <div class="err" style="margin-top:18px"><b>Error:</b> {{ error }}</div>
{% endif %}
{% if product %}
  <div class="result" style="margin-top:18px">
    <b>Resultados OCR:</b>
    <div>Producto: <b>{{ product }}</b></div>
    <div>Cantidad: <b>{{ quantity }}</b></div>
    <div style="margin-top:10px">
      {% if has_eml %}<a href="{{ url_for('download_eml', token=token) }}"><button class="secondary">Descargar .EML</button></a>{% endif %}
    </div>
  </div>
{% endif %}
</div></body></html>
"""

def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,11)
    th = cv2.medianBlur(th,3)
    return Image.fromarray(th)

def extract_product_and_qty(ocr_text: str):
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    pats = [
        re.compile(r"^(?P<qty>\d{1,3})\s*[xX×*]\s*(?P<prod>.+)$"),
        re.compile(r"^(?P<prod>.+?)\s*[xX×*]\s*(?P<qty>\d{1,3})$"),
        re.compile(r"^(?:Cant(?:idad)?|Qty|Cantidad)\s*[:=\-]?\s*(?P<qty>\d{1,3})\b.*(?P<prod>.+)$", re.I),
        re.compile(r"^(?P<prod>.+?)\s+(?:Cant(?:idad)?|Qty)\s*[:=\-]?\s*(?P<qty>\d{1,3})\b", re.I),
    ]
    for ln in lines:
        for p in pats:
            m = p.search(ln)
            if m:
                return m.group('prod').strip(), int(m.group('qty'))
    def looks_name(s): return len(s)>=5 and re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", s)
    for i, ln in enumerate(lines):
        if looks_name(ln):
            m2 = re.search(r"\b(\d{1,3})\b", ln)
            if m2: return ln, int(m2.group(1))
            for j in (i-1, i+1):
                if 0<=j<len(lines):
                    m3 = re.fullmatch(r"\s*(\d{1,3})\s*", lines[j])
                    if m3: return ln, int(m3.group(1))
    for ln in lines:
        if looks_name(ln): return ln, 1
    return "(No detectado)", 1

def build_eml_file(sender,to,subject,body_html,attachment_name,attachment_bytes):
    import email, email.policy
    from email.message import EmailMessage
    bio = BytesIO()
    msg = EmailMessage(policy=email.policy.default)
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    msg.set_content("Este mensaje requiere un cliente compatible con HTML.")
    msg.add_alternative(body_html, subtype="html")
    msg.add_attachment(attachment_bytes, maintype="image", subtype="png", filename=attachment_name)
    bio.write(msg.as_bytes()); bio.seek(0)
    return bio

SESS = {}
def save_blob(name, data):
    token = base64.urlsafe_b64encode(os.urandom(18)).decode('ascii').rstrip('=')
    SESS[token] = {"name": name, "data": data}
    return token
def get_blob(t): return SESS.get(t)

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files.get('image')
        to = request.form.get('to'); sender = request.form.get('sender')
        subject = request.form.get('subject') or 'Pedido detectado'
        note = request.form.get('note') or ''
        if not file or file.filename == '': return redirect(url_for('index'))

        img = Image.open(file.stream).convert('RGB')
        png_io = BytesIO(); img.save(png_io, format='PNG'); png_bytes = png_io.getvalue()

        pre = preprocess_for_ocr(img)
        # Fallbacks de idioma para evitar 500 si falta eng/spa
        ocr_text = None
        for lang in ('spa+eng','spa','eng',''):
            try:
                ocr_text = pytesseract.image_to_string(pre, lang=lang) if lang else pytesseract.image_to_string(pre)
                if (ocr_text or '').strip(): break
            except Exception as e:
                print(f"OCR falló con lang='{lang}':", e)
                continue
        if not ocr_text: ocr_text = ""

        product, qty = extract_product_and_qty(ocr_text)

        body_html = f"""
        <div style='font-family:Segoe UI,Arial,sans-serif;'>
          <p>Se detectó una compra a partir de la imagen cargada.</p>
          <ul><li><b>Producto:</b> {product}</li><li><b>Cantidad:</b> {qty}</li></ul>
          {('<p>' + (note.replace('\\n','<br>')) + '</p>') if note else ''}
          <p>Se adjunta la captura de pantalla original.</p>
        </div>
        """

        eml_stream = build_eml_file(sender, to, f"{subject}: {product} x{qty}", body_html, "captura.png", png_bytes)
        token = save_blob("correo.eml", eml_stream.getvalue())
        return render_template_string(INDEX_HTML, product=product, quantity=qty, token=token, has_eml=True)
    except Exception as e:
        # Log a consola y muestra mensaje al usuario
        print("ERROR /process:", e)
        traceback.print_exc()
        return render_template_string(INDEX_HTML, error=str(e)), 500

@app.route('/download.eml')
def download_eml():
    token = request.args.get('token','')
    blob = get_blob(token)
    if not blob: return "Token inválido", 404
    if not blob['name'].lower().endswith('.eml'): return "No hay .EML para este proceso", 400
    return send_file(BytesIO(blob['data']), mimetype='message/rfc822', as_attachment=True, download_name='correo.eml')

@app.get('/healthz')
def healthz(): return {"ok": True}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
PY

EXPOSE 8000
CMD ["sh","-c","gunicorn -w 2 -b 0.0.0.0:${PORT:-8000} app:app"]
