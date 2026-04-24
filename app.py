from flask import Flask, request, jsonify, render_template_string
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# ================= DATA =================
WORD_PAIRS = [
    ("happy","joyful",[0.9,0.5,0.8,0.3,0.5],1),
    ("happy","sad",[0.9,0.5,0.8,0.3,0.2],0),
    ("big","large",[0.5,0.6,0.9,0.2,0.4],1),
    ("big","small",[0.5,0.7,0.9,0.2,0.4],0),
    ("fast","quick",[0.5,0.6,0.8,0.2,0.4],1),
    ("fast","slow",[0.5,0.7,0.9,0.2,0.3],0),
    ("good","bad",[0.8,0.7,0.9,0.3,0.3],0),
    ("love","hate",[0.9,0.5,0.9,0.5,0.3],0),
]

X = np.array([p[2] for p in WORD_PAIRS])
y = np.array([p[3] for p in WORD_PAIRS])

WORD_FEATURES = {
    "happy":[0.9,0.6,0.85,0.3,0.45],
    "joyful":[0.9,0.5,0.8,0.3,0.5],
    "sad":[0.1,0.6,0.85,0.3,0.28],
    "big":[0.5,0.7,0.9,0.2,0.28],
    "large":[0.5,0.6,0.85,0.2,0.45],
    "small":[0.5,0.7,0.9,0.2,0.45],
    "fast":[0.5,0.7,0.85,0.2,0.38],
    "quick":[0.5,0.6,0.8,0.2,0.45],
    "slow":[0.5,0.7,0.85,0.2,0.38],
    "good":[0.8,0.7,0.95,0.3,0.38],
    "bad":[0.2,0.7,0.95,0.3,0.28],
    "love":[0.95,0.5,0.95,0.6,0.38],
    "hate":[0.05,0.5,0.85,0.5,0.38],
}

# ================= HTML =================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KNN – Phân Loại Từ Đồng/Trái Nghĩa</title>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}

body{
  background:#0a0a0f;
  color:#e8e8f0;
  font-family:'Sora',sans-serif;
  min-height:100vh;
  padding:40px 24px;
}

/* ---- HEADER ---- */
.header{text-align:center;margin-bottom:36px}

.badge{
  display:inline-block;
  background:#1a2a1e;
  color:#00ff88;
  font-size:11px;
  font-weight:600;
  letter-spacing:2px;
  padding:6px 18px;
  border-radius:999px;
  border:1px solid #00ff8840;
  margin-bottom:20px;
}

h1{
  font-size:clamp(28px,5vw,48px);
  font-weight:800;
  line-height:1.15;
  margin-bottom:8px;
}

h1 span{color:#00ff88}

.sub{
  font-size:13px;
  color:#555;
  letter-spacing:.5px;
}

/* ---- CARD ---- */
.card{
  background:#1a1a24;
  border:1px solid #2a2a3a;
  border-radius:16px;
  padding:28px;
  max-width:700px;
  margin:0 auto 20px;
}

/* ---- INPUTS ---- */
.row{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:16px;
  margin-bottom:20px;
}

.field label{
  display:block;
  font-size:10px;
  font-weight:600;
  letter-spacing:2px;
  color:#00ff88;
  margin-bottom:8px;
}

.field input[type=text]{
  width:100%;
  background:#0f0f18;
  border:1px solid #2a2a3a;
  border-radius:10px;
  padding:14px 16px;
  color:#fff;
  font-size:15px;
  font-family:'Sora',sans-serif;
  outline:none;
  transition:border .2s;
}

.field input[type=text]:focus{border-color:#00ff88}

/* ---- SLIDER ---- */
.slider-row{
  display:flex;
  align-items:center;
  gap:14px;
  margin-bottom:24px;
}

.slider-row label{
  font-size:11px;
  font-weight:600;
  letter-spacing:1.5px;
  color:#888;
  white-space:nowrap;
}

.slider-row input[type=range]{
  flex:1;
  accent-color:#00ff88;
}

.kval{
  font-size:14px;
  font-weight:700;
  color:#fff;
  min-width:14px;
}

.kdesc{font-size:12px;color:#555}

/* ---- BUTTON ---- */
.btn{
  width:100%;
  background:#00ff88;
  color:#050f07;
  border:none;
  padding:15px;
  border-radius:12px;
  font-size:14px;
  font-weight:800;
  letter-spacing:1.5px;
  cursor:pointer;
  font-family:'Sora',sans-serif;
  transition:box-shadow .2s, transform .1s;
}

.btn:hover{box-shadow:0 0 28px #00ff8855}
.btn:active{transform:scale(0.98)}

/* ---- RESULT ---- */
.result-card{
  background:#1a1a24;
  border:1px solid #2a2a3a;
  border-radius:16px;
  padding:28px;
  max-width:700px;
  margin:0 auto 20px;
  display:none;
}

.result-inner{
  border-radius:12px;
  padding:24px;
  text-align:center;
}

.result-inner.syn{border:1px solid #00ff88;background:#0a1a0f}
.result-inner.ant{border:1px solid #ff6b6b;background:#1a0a0a}

.result-tag{
  font-size:11px;
  font-weight:700;
  letter-spacing:2px;
  margin-bottom:12px;
}

.result-tag.syn{color:#00ff88}
.result-tag.ant{color:#ff6b6b}

.result-words{
  font-size:22px;
  font-weight:800;
  margin-bottom:20px;
}

/* ---- BARS ---- */
.bar-wrap{margin-bottom:10px}

.bar-label{
  display:flex;
  justify-content:space-between;
  font-size:12px;
  color:#888;
  margin-bottom:5px;
}

.bar-bg{
  background:#ffffff12;
  border-radius:999px;
  height:8px;
  overflow:hidden;
}

.bar-fill{
  height:100%;
  border-radius:999px;
  width:0%;
  transition:width .6s cubic-bezier(.4,0,.2,1);
}

.bar-fill.syn{background:#00ff88}
.bar-fill.ant{background:#ff6b6b}

/* ---- VOCAB ---- */
.vocab-card{
  background:#1a1a24;
  border:1px solid #2a2a3a;
  border-radius:16px;
  padding:24px;
  max-width:700px;
  margin:0 auto;
}

.vocab-title{
  font-size:10px;
  font-weight:700;
  letter-spacing:2px;
  color:#555;
  margin-bottom:16px;
  display:flex;
  align-items:center;
  gap:8px;
}

.vocab-title::before{
  content:'';
  display:inline-block;
  width:8px;height:8px;
  border-radius:50%;
  background:#00ff88;
}

.vocab-chips{display:flex;flex-wrap:wrap;gap:8px}

.chip{
  background:#0f0f18;
  border:1px solid #2a2a3a;
  border-radius:8px;
  padding:6px 14px;
  font-size:13px;
  color:#aaa;
  cursor:pointer;
  font-family:'Sora',sans-serif;
  transition:border .2s, color .2s;
}

.chip:hover{border-color:#00ff88;color:#00ff88}
</style>
</head>
<body>

<div class="header">
  <div class="badge">THUẬT TOÁN KNN</div>
  <h1>Phân Loại Từ<br><span>Đồng &amp; Trái Nghĩa</span></h1>
  <p class="sub">K-Nearest Neighbors · Tiếng Anh · Scikit-learn</p>
</div>

<!-- Input card -->
<div class="card">
  <div class="row">
    <div class="field">
      <label>TỪ GỐC (WORD 1)</label>
      <input type="text" id="word1" placeholder="happy">
    </div>
    <div class="field">
      <label>TỪ SO SÁNH (WORD 2)</label>
      <input type="text" id="word2" placeholder="joyful">
    </div>
  </div>

  <div class="slider-row">
    <label>K =</label>
    <input type="range" id="kval" min="1" max="7" step="2" value="3">
    <span class="kval" id="kshow">3</span>
    <span class="kdesc">láng giềng gần nhất</span>
  </div>

  <button class="btn" onclick="predict()">PHÂN TÍCH →</button>
</div>

<!-- Result card -->
<div class="result-card" id="resultCard">
  <div class="result-inner" id="resultInner">
    <div class="result-tag" id="rtag"></div>
    <div class="result-words" id="rwords"></div>
    <div class="bar-wrap">
      <div class="bar-label"><span>Đồng nghĩa</span><span id="pSyn">-</span></div>
      <div class="bar-bg"><div class="bar-fill syn" id="barSyn"></div></div>
    </div>
    <div class="bar-wrap" style="margin-top:10px">
      <div class="bar-label"><span>Trái nghĩa</span><span id="pAnt">-</span></div>
      <div class="bar-bg"><div class="bar-fill ant" id="barAnt"></div></div>
    </div>
  </div>
</div>

<!-- Vocab card -->
<div class="vocab-card">
  <div class="vocab-title">TỪ CÓ SẴN TRONG HỆ THỐNG</div>
  <div class="vocab-chips" id="chips"></div>
</div>

<script>
// Slider label
document.getElementById('kval').oninput = function(){
  document.getElementById('kshow').textContent = this.value;
};

// Build word chips (click to fill inputs)
const WORDS = {{ words|tojson }};
const chipsEl = document.getElementById('chips');
WORDS.forEach(w => {
  const c = document.createElement('button');
  c.className = 'chip';
  c.textContent = w;
  c.onclick = () => {
    const i1 = document.getElementById('word1');
    const i2 = document.getElementById('word2');
    if (!i1.value) i1.value = w;
    else if (!i2.value) i2.value = w;
  };
  chipsEl.appendChild(c);
});

// Call Flask /predict
async function predict(){
  const word1 = document.getElementById('word1').value.trim().toLowerCase();
  const word2 = document.getElementById('word2').value.trim().toLowerCase();
  const k     = parseInt(document.getElementById('kval').value);

  if(!word1 || !word2){ alert('Nhập đủ 2 từ!'); return; }

  const rc = document.getElementById('resultCard');
  rc.style.display = 'block';
  document.getElementById('rtag').textContent = 'Đang phân tích...';
  document.getElementById('rwords').textContent = '';
  document.getElementById('barSyn').style.width = '0%';
  document.getElementById('barAnt').style.width = '0%';

  try{
    const resp = await fetch('/predict',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({word1, word2, k})
    });
    const data = await resp.json();

    if(data.error){
      document.getElementById('resultInner').className = 'result-inner ant';
      document.getElementById('rtag').className = 'result-tag ant';
      document.getElementById('rtag').textContent = '✗ LỖI';
      document.getElementById('rwords').textContent = data.error;
      return;
    }

    const isSyn = data.prediction === 1;
    const synPct = (data.probabilities[1] * 100).toFixed(1);
    const antPct = (data.probabilities[0] * 100).toFixed(1);

    document.getElementById('resultInner').className = 'result-inner ' + (isSyn ? 'syn' : 'ant');
    document.getElementById('rtag').className = 'result-tag ' + (isSyn ? 'syn' : 'ant');
    document.getElementById('rtag').textContent   = isSyn ? '✓ ĐỒNG NGHĨA' : '✗ TRÁI NGHĨA';
    document.getElementById('rwords').textContent = word1 + ' — ' + word2;
    document.getElementById('pSyn').textContent   = synPct + '%';
    document.getElementById('pAnt').textContent   = antPct + '%';

    setTimeout(() => {
      document.getElementById('barSyn').style.width = synPct + '%';
      document.getElementById('barAnt').style.width = antPct + '%';
    }, 50);

  }catch(e){
    document.getElementById('rtag').textContent = 'Lỗi kết nối!';
  }
}
</script>

</body>
</html>
"""

# ================= ROUTES =================
@app.route('/')
def index():
    words = list(WORD_FEATURES.keys())
    return render_template_string(HTML_TEMPLATE, words=words)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data  = request.get_json(force=True)
        word1 = data.get('word1','').lower()
        word2 = data.get('word2','').lower()
        k     = int(data.get('k', 3))

        if word1 not in WORD_FEATURES:
            return jsonify({'error': f'Từ "{word1}" không có trong hệ thống'})
        if word2 not in WORD_FEATURES:
            return jsonify({'error': f'Từ "{word2}" không có trong hệ thống'})
        if word1 == word2:
            return jsonify({'error': '2 từ giống nhau'})

        f1    = np.array(WORD_FEATURES[word1])
        f2    = np.array(WORD_FEATURES[word2])
        query = ((f1 + f2) / 2).reshape(1, -1)

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)

        pred = int(model.predict(query)[0])
        prob = model.predict_proba(query)[0]

        return jsonify({
            'prediction':   pred,
            'probabilities': prob.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True)
    