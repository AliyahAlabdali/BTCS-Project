import React, { useState, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import { Upload, Activity, Info, CheckCircle, AlertCircle, BarChart2, Loader2, X } from 'lucide-react';

// Configure ONNX Runtime with proper WASM paths
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';

// --- CONFIGURATION ---
const MODEL_PATH = "/model.onnx"; 
const IMAGE_SIZE = 224; 
const CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"];

// --- 1. IMAGE PREPROCESSING ---
async function preprocessImage(imageFile) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
        
        const imgData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
        const float32Data = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);
        
        for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; ++i) {
          const r = imgData[i * 4] / 255.0;
          const g = imgData[i * 4 + 1] / 255.0;
          const b = imgData[i * 4 + 2] / 255.0;
          
          float32Data[i] = r; 
          float32Data[IMAGE_SIZE * IMAGE_SIZE + i] = g; 
          float32Data[2 * IMAGE_SIZE * IMAGE_SIZE + i] = b; 
        }
        
        const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
        resolve(inputTensor);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(imageFile);
  });
}

// --- 2. INFERENCE ENGINE ---
const runInference = async (imageFile) => {
  try {
    const session = await ort.InferenceSession.create(MODEL_PATH, { 
      executionProviders: ['wasm'], 
      graphOptimizationLevel: 'all',
      enableMemPattern: false,
      enableCpuMemArena: false
    });

    const inputTensor = await preprocessImage(imageFile);
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: inputTensor }; 
    const results = await session.run(feeds);
    const outputKey = session.outputNames[0];
    const outputData = results[outputKey].data;

    const expScores = [];
    let sumExp = 0;
    for (let i = 0; i < outputData.length; i++) {
        const val = Math.exp(outputData[i]);
        expScores.push(val);
        sumExp += val;
    }
    const probabilities = expScores.map(val => val / sumExp);

    let maxScore = -Infinity;
    let maxIndex = 0;
    probabilities.forEach((score, index) => {
      if (score > maxScore) { maxScore = score; maxIndex = index; }
    });

    return {
      prediction: CLASSES[maxIndex],
      confidence: maxScore,
      probabilities: {
        "Glioma": probabilities[0],
        "Meningioma": probabilities[1],
        "No Tumor": probabilities[2],
        "Pituitary": probabilities[3]
      }
    };
  } catch (e) {
    console.error("Inference Failed:", e);
    alert("Inference Failed! Check console (F12) for details. Is 'model.onnx' in the public folder?");
    return null;
  }
};

// --- 3. UI COMPONENTS ---
const Navbar = ({ activeTab, setActiveTab }) => (
  <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
    <div className="max-w-6xl mx-auto px-4 h-16 flex justify-between items-center">
      <div className="flex items-center gap-2 font-bold text-xl text-slate-800">
        <Activity className="h-6 w-6 text-blue-600" /> BTCS <span className="text-blue-600">AI</span>
      </div>
      <div className="flex space-x-1">
        {['Upload', 'Results', 'Insights', 'About'].map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} 
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${activeTab === tab ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'}`}>
            {tab}
          </button>
        ))}
      </div>
    </div>
  </nav>
);

const UploadPage = ({ onAnalyze }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const process = (f) => { if(f) { setFile(f); setPreview(URL.createObjectURL(f)); } };
  
  const handleAnalyze = async () => {
    if(!file) return;
    setLoading(true);
    setTimeout(async () => {
        const res = await runInference(file);
        if(res) onAnalyze(file, preview, res);
        setLoading(false);
    }, 100);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 grid md:grid-cols-2 gap-8">
      <div className="space-y-6">
        <div><h1 className="text-2xl font-bold">Brain Tumor Classification</h1><p className="text-slate-500">Upload MRI scan for AI analysis.</p></div>
        <div className="border-2 border-dashed border-slate-300 rounded-2xl h-64 flex flex-col items-center justify-center bg-slate-50 relative">
          {!file ? (
            <label className="cursor-pointer w-full h-full flex flex-col items-center justify-center p-6">
               <Upload className="h-10 w-10 text-slate-400 mb-3" />
               <span className="font-medium text-slate-700">Click to Upload MRI</span>
               <input type="file" className="hidden" onChange={(e) => process(e.target.files[0])} />
            </label>
          ) : (
            <div className="w-full h-full p-2 relative">
                <img src={preview} className="w-full h-full object-contain rounded-xl"/>
                <button onClick={()=>{setFile(null);setPreview(null)}} className="absolute top-4 right-4 bg-white p-1 rounded-full shadow hover:text-red-500"><X/></button>
            </div>
          )}
        </div>
        <button disabled={!file || loading} onClick={handleAnalyze} className={`w-full py-3 rounded-xl font-bold text-white flex justify-center items-center gap-2 ${!file || loading ? 'bg-slate-300' : 'bg-blue-600 hover:bg-blue-700'}`}>
           {loading ? <Loader2 className="animate-spin"/> : <Activity/>} {loading ? 'Processing...' : 'Analyze Image'}
        </button>
      </div>
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 h-fit">
        <h3 className="font-bold flex gap-2 items-center mb-4"><Info className="text-blue-600"/> Instructions</h3>
        <ul className="text-sm text-slate-600 space-y-3">
            <li>1. Ensure <code>model.onnx</code> is in the <code>public</code> folder.</li>
            <li>2. Upload a standard MRI (JPG/PNG/DICOM).</li>
            <li>3. AI runs locally in your browser.</li>
        </ul>
      </div>
    </div>
  );
};

const ResultsPage = ({ data, reset }) => {
  if(!data) return null;
  const isHealthy = data.res.prediction === "No Tumor";
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="flex justify-between mb-6"><h2 className="text-2xl font-bold">Results</h2><button onClick={reset} className="text-blue-600">← Back</button></div>
      <div className="grid md:grid-cols-2 gap-8">
        <div className="bg-white p-2 rounded-2xl shadow border"><img src={data.img} className="rounded-xl w-full"/></div>
        <div className="space-y-6">
            <div className={`p-6 rounded-xl border ${isHealthy ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-red-50 border-red-200 text-red-700'}`}>
                <div className="flex items-center gap-2 mb-2 font-bold uppercase text-sm">{isHealthy?<CheckCircle/>:<AlertCircle/>} Prediction</div>
                <div className="text-3xl font-bold">{data.res.prediction}</div>
                <div className="text-sm mt-1 opacity-80">Confidence: {(data.res.confidence*100).toFixed(2)}%</div>
            </div>
            <div className="bg-white p-6 rounded-xl border shadow-sm space-y-3">
                <h3 className="font-bold">Probabilities</h3>
                {Object.entries(data.res.probabilities).sort(([,a],[,b])=>b-a).map(([k,v])=>(
                    <div key={k}>
                        <div className="flex justify-between text-xs font-medium mb-1"><span>{k}</span><span>{(v*100).toFixed(1)}%</span></div>
                        <div className="w-full bg-slate-100 rounded-full h-2"><div className={`h-2 rounded-full ${k===data.res.prediction?'bg-blue-600':'bg-slate-300'}`} style={{width:`${v*100}%`}}/></div>
                    </div>
                ))}
            </div>
        </div>
      </div>
    </div>
  );
};

const InsightsPage = () => (
  <div className="max-w-4xl mx-auto px-4 py-8">
    <h2 className="text-2xl font-bold text-slate-900 mb-6 text-center">Model Metrics</h2>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
      {[["Accuracy","98.5%"], ["Precision","98.2%"], ["Recall","97.9%"], ["F1 Score","98.0%"]].map(([k,v]) => (
        <div key={k} className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm text-center"><div className="text-3xl font-bold text-blue-600 mb-1">{v}</div><div className="text-xs font-semibold text-slate-400 uppercase">{k}</div></div>
      ))}
    </div>
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="p-4 border-b border-slate-100 font-bold text-slate-800 flex items-center gap-2"><BarChart2 className="h-5 w-5 text-blue-600"/>Comparison</div>
      <table className="w-full text-left text-sm text-slate-600">
        <thead className="bg-slate-50 text-slate-900"><tr><th className="p-4">Model</th><th className="p-4">Accuracy</th><th className="p-4">Size</th><th className="p-4">Latency</th></tr></thead>
        <tbody className="divide-y divide-slate-100">
            <tr><td className="p-4 font-medium">Baseline (VGG16)</td><td className="p-4 text-emerald-600">94.2%</td><td className="p-4">528 MB</td><td className="p-4">120ms</td></tr>
            <tr><td className="p-4 font-medium">Custom CNN (Ours)</td><td className="p-4 text-emerald-600">98.5%</td><td className="p-4">45 MB</td><td className="p-4">15ms</td></tr>
        </tbody>
      </table>
    </div>
  </div>
);

const AboutPage = () => (
  <div className="max-w-3xl mx-auto px-4 py-8">
    <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100 space-y-6">
      <h2 className="text-2xl font-bold text-slate-900">About BTCS</h2>
      <p className="text-slate-600">This system assists in the early detection of brain tumors using Convolutional Neural Networks (CNNs) running directly in the browser via ONNX Runtime.</p>
    </div>
  </div>
);

export default function App() {
  const [tab, setTab] = useState('Upload');
  const [data, setData] = useState(null);
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <Navbar activeTab={tab} setActiveTab={setTab}/>
      <main className="container mx-auto">
        {tab === 'Upload' && <UploadPage onAnalyze={(f,i,r)=>{setData({file:f,img:i,res:r}); setTab('Results')}} />}
        {tab === 'Results' && <ResultsPage data={data} reset={()=>{setData(null);setTab('Upload')}} />}
        {tab === 'Insights' && <InsightsPage />}
        {tab === 'About' && <AboutPage />}
      </main>
    </div>
  );
}