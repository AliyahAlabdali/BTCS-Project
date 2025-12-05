# Brain Tumor Classification System (BTCS)

## 🧠 AI System Design Course Project

An intelligent web-based diagnostic tool that leverages deep learning to classify brain tumors from MRI scans. The system runs entirely in the browser using ONNX Runtime, ensuring privacy and accessibility.

## ✨ Features

- **🎯 Real-time AI Classification**: Instant brain tumor classification with confidence scores
- **🔒 Privacy-First**: All processing happens locally in your browser - no data is sent to servers
- **⚡ Fast Processing**: CNN model optimized for quick inference (15ms latency)
- **📊 Detailed Analytics**: Probability distribution across all tumor types
- **🎨 Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS
- **📱 Mobile Friendly**: Works seamlessly on desktop, tablet, and mobile devices

## Tumor Classification Types

The system can classify four types of brain tumors:

1. **Glioma** - A tumor in glial cells of the brain and spinal cord
2. **Meningioma** - A tumor in the membranes surrounding the brain
3. **Pituitary** - An abnormal growth in the pituitary gland
4. **No Tumor** - Healthy brain tissue with no abnormalities

## 🚀 Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Ensure `model.onnx` is in the `public/` folder


## 📋 How to Use

1. **Upload MRI Scan**: Click or drag an MRI scan image (JPG/PNG format) into the upload box
2. **Run Analysis**: Click the "Run Analysis" button
3. **View Results**: Get instant classification results with probability scores
4. **Analyze Another**: Click "Analyze Another" to process a new image

## 🏗️ Technology Stack

- **Frontend**: React 19, JavaScript, TailwindCSS
- **AI/ML**: ONNX Runtime Web, CNN Model
- **Build Tool**: Vite 7
- **Icons**: Lucide React
- **Styling**: PostCSS, Autoprefixer

## 📁 Project Structure

```
btcs-app/
├── src/
│   ├── App.jsx              # Main application component
│   ├── App.css              # Application styles
│   ├── main.jsx             # Entry point
│   ├── index.css            # Global styles
│   └── assets/              # Images and assets
├── public/
│   ├── model.onnx           # Pre-trained CNN model
│   └── ort-wasm-*.wasm      # ONNX Runtime WASM files
├── index.html               # HTML template
├── vite.config.js           # Vite configuration
├── tailwind.config.js       # Tailwind CSS config
└── package.json             # Project dependencies
```

## 🤖 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 224x224 RGB images
- **Output**: 4 class probabilities (Glioma, Meningioma, Pituitary, No Tumor)
- **Accuracy**: ~79%
- **Model Size**: ~1.45 MB
- **Inference Time**: ~9ms

## ⚖️ Disclaimer

⚠️ **For Educational and Research Use Only**

This AI system is designed for educational purposes and should NOT be used as a primary diagnostic tool. Always consult a certified radiologist for medical diagnosis. The accuracy of predictions depends on image quality and input data.

## 📚 Development

### Code Quality

The project uses ESLint for code quality and follows React best practices.

## 🛠️ Troubleshooting

### Model not loading?
- Ensure `model.onnx` is in the `public/` folder
- Check browser console (F12) for detailed error messages
- Clear browser cache and reload

### Slow inference?
- The first inference may be slower due to WASM initialization
- Subsequent inferences should be faster
- Check your browser's hardware acceleration settings

## 👤 Authors

**Aliyah Alabdali**
**Renad Alharthi**
**Rahaf Almalki**
**Rawan Alahmadi**

## 📄 License

This project is provided for educational purposes.

**Last Updated**: December 2025

**Status**: ✅ Final Version - Production Ready
