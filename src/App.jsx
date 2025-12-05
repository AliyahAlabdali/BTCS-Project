import React, { useState, useEffect } from 'react';

import * as ort from 'onnxruntime-web';

import { Upload, Activity, Info, CheckCircle, AlertCircle, Loader2, X, Brain, Shield, ChevronRight } from 'lucide-react';



// --- CONFIGURATION ---

const MODEL_PATH = "./model.onnx"; 

const IMAGE_SIZE = 224; 

const CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"];

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
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
    const session = await ort.InferenceSession.create(MODEL_PATH, { 
      executionProviders: ['wasm'], 
      graphOptimizationLevel: 'all' 
    });

    const inputTensor = await preprocessImage(imageFile);
    const feeds = { input: inputTensor }; 
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
        "Pituitary": probabilities[2],
        "No Tumor": probabilities[3]
      }
    };
  } catch (e) {
    console.error("Inference Failed:", e);
    alert("Analysis Error. Please ensure 'model.onnx' is loaded correctly.");
    return null;
  }
};

// --- 3. UI COMPONENTS ---

const Navbar = ({ activeTab, setActiveTab }) => (
  <nav className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
    <div className="max-w-6xl mx-auto px-4 h-16 flex justify-between items-center">
      <div className="flex items-center gap-2 font-bold text-xl text-slate-800 tracking-tight cursor-pointer" onClick={() => setActiveTab('Upload')}>
        <div className="bg-blue-600 p-1.5 rounded-lg text-white">
            <Brain className="h-5 w-5" />
        </div>
        <span>BTCS <span className="text-blue-600">AI</span></span>
      </div>
      <div className="hidden md:flex space-x-1">
        {['Upload', 'Results', 'About'].map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} 
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${activeTab === tab ? 'bg-blue-50 text-blue-700 shadow-sm' : 'text-slate-600 hover:bg-slate-50'}`}>
            {tab}
          </button>
        ))}
      </div>
    </div>
  </nav>
);

const Footer = () => (
  <footer className="bg-white border-t border-slate-200 mt-auto">
    <div className="max-w-6xl mx-auto px-4 py-6 flex flex-col md:flex-row justify-between items-center text-sm text-slate-500">
      <div className="flex items-center gap-2">
        <Shield className="h-4 w-4 text-slate-400" />
        <span>Graduation Project 2025</span>
      </div>
      <div className="mt-2 md:mt-0">
        AI-Powered Medical Diagnostics • <span className="text-blue-600 font-medium">For Research Use Only</span>
      </div>
    </div>
  </footer>
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
    }, 500);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 animate-fade-in">
      <div className="text-center mb-10 space-y-2">
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">AI-Powered Brain Tumor Detection</h1>
        <p className="text-lg text-slate-500 max-w-2xl mx-auto">
            A rapid, privacy-focused diagnostic tool assisting medical professionals in the early classification of brain tumors using Deep Learning.
        </p>
      </div>

      <div className="grid md:grid-cols-12 gap-8">
        {/* Upload Card */}
        <div className="md:col-span-7 space-y-6">
          <div className={`border-2 border-dashed rounded-3xl h-80 flex flex-col items-center justify-center transition-all duration-300 relative overflow-hidden group
            ${file ? 'border-blue-500 bg-blue-50/30' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'}`}>
            
            {!file ? (
              <label className="cursor-pointer w-full h-full flex flex-col items-center justify-center p-6 z-10">
                 <div className="bg-white p-4 rounded-full shadow-sm mb-4 group-hover:scale-110 transition-transform">
                    <Upload className="h-8 w-8 text-blue-600" />
                 </div>
                 <span className="font-semibold text-lg text-slate-700">Drag & Drop MRI Scan</span>
                 <span className="text-sm text-slate-400 mt-2">Supports JPG, PNG, DICOM</span>
                 <input type="file" className="hidden" onChange={(e) => process(e.target.files[0])} />
              </label>
            ) : (
              <div className="w-full h-full p-4 relative flex items-center justify-center">
                  <img src={preview} className="max-h-full max-w-full object-contain rounded-xl shadow-sm"/>
                  <button onClick={()=>{setFile(null);setPreview(null)}} className="absolute top-4 right-4 bg-white/90 p-2 rounded-full shadow-md hover:bg-red-50 text-slate-600 hover:text-red-500 transition-colors">
                    <X className="h-5 w-5"/>
                  </button>
              </div>
            )}
          </div>

          <button disabled={!file || loading} onClick={handleAnalyze} 
            className={`w-full py-4 rounded-xl font-bold text-white text-lg shadow-lg hover:shadow-xl transition-all flex justify-center items-center gap-3
            ${!file || loading ? 'bg-slate-300 cursor-not-allowed' : 'bg-gradient-to-r from-blue-600 to-blue-700 hover:scale-[1.01]'}`}>
             {loading ? <Loader2 className="animate-spin h-6 w-6"/> : <Activity className="h-6 w-6"/>} 
             {loading ? 'Analyzing Scan...' : 'Run Analysis'}
          </button>
        </div>

        {/* Instructions Card */}
        <div className="md:col-span-5">
          <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100 h-full">
            <h3 className="font-bold text-lg flex gap-2 items-center mb-6 text-slate-800">
                <Info className="text-blue-600 h-5 w-5"/> How to Use
            </h3>
            <div className="space-y-6">
                {[
                    {step: 1, title: "Prepare Image", desc: "Ensure you have a clear T1-weighted MRI scan in JPG or PNG format."},
                    {step: 2, title: "Upload Scan", desc: "Drag the file into the upload box on the left."},
                    {step: 3, title: "AI Analysis", desc: "Click 'Run Analysis'. The CNN model processes the image locally in your browser."},
                    {step: 4, title: "View Report", desc: "Instant classification results will appear with confidence scores."}
                ].map((item) => (
                    <div key={item.step} className="flex gap-4">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-sm">
                            {item.step}
                        </div>
                        <div>
                            <div className="font-semibold text-slate-800 text-sm">{item.title}</div>
                            <div className="text-xs text-slate-500 leading-relaxed mt-1">{item.desc}</div>
                        </div>
                    </div>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ResultsPage = ({ data, reset }) => {
  if(!data) return null;
  const isHealthy = data.res.prediction === "No Tumor";
  
  return (
    <div className="max-w-6xl mx-auto px-4 py-10 animate-fade-in">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold text-slate-900">Analysis Report</h2>
        <button onClick={reset} className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors">
            ← Analyze Another
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Left Column */}
        <div className="space-y-6">
            <div className="bg-black/5 p-2 rounded-2xl border border-slate-200 shadow-inner flex items-center justify-center h-96">
                <img src={data.img} className="max-h-full rounded-xl object-contain"/>
            </div>
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Disclaimer</h3>
                <p className="text-slate-700 text-sm leading-relaxed">
                    This AI result is for educational and experimental purposes only. It should not be used as a primary diagnostic tool. Always consult a certified radiologist.
                </p>
            </div>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
            <div className={`p-8 rounded-2xl border-l-8 shadow-sm flex items-start gap-4
                ${isHealthy ? 'bg-emerald-50 border-emerald-500' : 'bg-red-50 border-red-500'}`}>
                <div className={`p-3 rounded-full ${isHealthy ? 'bg-emerald-100 text-emerald-600' : 'bg-red-100 text-red-600'}`}>
                    {isHealthy ? <CheckCircle className="h-8 w-8"/> : <AlertCircle className="h-8 w-8"/>}
                </div>
                <div>
                    <div className={`text-sm font-bold uppercase tracking-wide mb-1 ${isHealthy ? 'text-emerald-700' : 'text-red-700'}`}>
                        Detected Class
                    </div>
                    <div className="text-4xl font-extrabold text-slate-900 mb-2">{data.res.prediction}</div>
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/60 text-sm font-medium text-slate-700 border border-slate-200">
                        <Activity className="h-3 w-3 text-blue-500"/>
                        Confidence: {(data.res.confidence*100).toFixed(2)}%
                    </div>
                </div>
            </div>

            <div className="bg-white p-8 rounded-2xl border border-slate-100 shadow-lg">
                <h3 className="font-bold text-slate-800 mb-6 flex items-center gap-2">
                    <Activity className="text-blue-600"/> Detailed Probabilities
                </h3>
                <div className="space-y-5">
                    {Object.entries(data.res.probabilities).sort(([,a],[,b])=>b-a).map(([k,v])=>(
                        <div key={k}>
                            <div className="flex justify-between text-sm font-medium mb-2 text-slate-700">
                                <span>{k}</span>
                                <span className="text-slate-500">{(v*100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                                <div 
                                    className={`h-full rounded-full transition-all duration-1000 ease-out ${k===data.res.prediction ? (isHealthy ? 'bg-emerald-500' : 'bg-blue-600') : 'bg-slate-300'}`} 
                                    style={{width:`${v*100}%`}}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
      </div>
    </div>
  );
};

const AboutPage = () => (
  <div className="max-w-4xl mx-auto px-4 py-12 animate-fade-in">
    <div className="bg-white rounded-3xl shadow-sm border border-slate-100 overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-900 to-slate-800 p-10 text-white">
            <h2 className="text-3xl font-bold mb-4">About the System</h2>
            <p className="text-slate-300 leading-relaxed text-lg max-w-2xl">
                The Brain Tumor Classification System (BTCS) is a graduation project dedicated to leveraging Artificial Intelligence for healthcare advancement.
            </p>
        </div>
        
        <div className="p-10 space-y-10">
            {/* Tumor Guide (NEW ENHANCEMENT) */}
            <section>
                <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                    <Brain className="text-blue-600"/> Tumor Classification Guide
                </h3>
                <div className="grid sm:grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl bg-orange-50 border border-orange-100">
                        <div className="font-bold text-orange-800 mb-1">Glioma</div>
                        <div className="text-sm text-slate-600">A tumor occurring in the brain and spinal cord, arising from glial cells.</div>
                    </div>
                    <div className="p-4 rounded-xl bg-purple-50 border border-purple-100">
                        <div className="font-bold text-purple-800 mb-1">Meningioma</div>
                        <div className="text-sm text-slate-600">A tumor that forms on the membranes (meninges) covering the brain.</div>
                    </div>
                    <div className="p-4 rounded-xl bg-blue-50 border border-blue-100">
                        <div className="font-bold text-blue-800 mb-1">Pituitary</div>
                        <div className="text-sm text-slate-600">An abnormal growth in the pituitary gland, affecting hormone balance.</div>
                    </div>
                    <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-100">
                        <div className="font-bold text-emerald-800 mb-1">No Tumor</div>
                        <div className="text-sm text-slate-600">Healthy brain tissue with no detectable abnormalities.</div>
                    </div>
                </div>
            </section>
            
            <div className="border-t border-slate-100 pt-8">
                <h3 className="text-xl font-bold text-slate-900 mb-4">Technology Stack</h3>
                <div className="flex flex-wrap gap-2">
                    {['React', 'TailwindCSS', 'ONNX Runtime', 'Python', 'CNN'].map(tag => (
                        <span key={tag} className="px-3 py-1 rounded-full bg-slate-100 text-slate-600 text-xs font-medium">
                            {tag}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    </div>
  </div>
);

export default function App() {
  const [tab, setTab] = useState('Upload');
  const [data, setData] = useState(null);
  
  return (
    <div className="min-h-screen bg-slate-50/50 font-sans text-slate-900 flex flex-col">
      <Navbar activeTab={tab} setActiveTab={setTab}/>
      <main className="container mx-auto pb-10 flex-grow">
        {tab === 'Upload' && <UploadPage onAnalyze={(f,i,r)=>{setData({file:f,img:i,res:r}); setTab('Results')}} />}
        {tab === 'Results' && <ResultsPage data={data} reset={()=>{setData(null);setTab('Upload')}} />}
        {tab === 'About' && <AboutPage />}
      </main>
      <Footer />
    </div>
  );
}