Comment Toxicity Detection 🔍

An AI-powered machine learning model that automatically detects **6 types of toxic comments** using Deep Learning (LSTM Neural Networks).

📋 Project Overview

This project builds a toxicity detection system that can identify harmful comments across multiple dimensions:
- **Toxic** - Generally abusive comments
- **Severe Toxic** - Extremely abusive language
- **Obscene** - Profanity and vulgar language
- **Threat** - Direct threats of violence
- **Insult** - Personal attacks and insults
- **Identity Hate** - Attacks based on identity

🎯 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.4% |
| **Precision** | 80.8% |
| **Recall** | 74.1% |
| **Training Epochs** | 3 |

🛠️ Technologies Used

- **Language:** Python 3.11
- **ML Framework:** TensorFlow 2.20, Keras
- **Web Interface:** Gradio
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **ML Algorithms:** Bidirectional LSTM, Neural Networks

📊 Model Architecture
```
Input Comments (Text)
        ↓
TextVectorization Layer (200,000 vocab, 1800 length)
        ↓
Embedding Layer (32-dimensional word embeddings)
        ↓
Bidirectional LSTM (32 units, tanh activation)
        ↓
Dense Layer (128 neurons, relu activation)
        ↓
Dense Layer (256 neurons, relu activation)
        ↓
Dense Layer (128 neurons, relu activation)
        ↓
Output Layer (6 neurons, sigmoid activation)
        ↓
6 Toxicity Predictions (0.0 - 1.0 confidence)
```

📁 Dataset

- **Source:** Jigsaw Toxic Comment Classification Challenge
- **Size:** 160,000 labeled comments
- **Training Set:** 70% (112,000 comments)
- **Validation Set:** 20% (32,000 comments)
- **Test Set:** 10% (16,000 comments)

**Download Dataset:**
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

🚀 Installation

Requirements
- Python 3.11+
- pip (Python package manager)

Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Comment-Toxicity-Detection.git
cd Comment-Toxicity-Detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run Jupyter Notebook:**
```bash
jupyter notebook Comment_toxicity.ipynb
```

 Usage

Running the Model

1. Open `Comment_toxicity.ipynb` in Jupyter
2. Run all cells in order
3. Scroll to the Gradio interface at the bottom
4. A web interface will launch automatically

Testing the Model

**Example 1 - Toxic Comment:**
```
Input: "You are so stupid and dumb"

Output:
✓ Toxic: YES (Confidence: 99.85%)
✓ Obscene: YES (Confidence: 98.14%)
✓ Insult: YES (Confidence: 92.24%)
✗ Threat: NO (Confidence: 13.70%)
✗ Severe Toxic: NO (Confidence: 48.59%)
✗ Identity Hate: NO (Confidence: 30.98%)
```

**Example 2 - Non-Toxic Comment:**
```
Input: "This is a great day!"

Output:
✗ All toxicity types: NO (Low confidence scores)
```

🎨 Features

✅ **Real-time Detection** - Instant toxicity analysis  
✅ **Multi-label Classification** - Detects multiple toxicity types at once  
✅ **Confidence Scores** - Shows probability for each toxicity type  
✅ **Visual Dashboard** - Colorful bar charts (red=toxic, green=safe)  
✅ **Web Interface** - Easy-to-use Gradio interface  
✅ **Professional UI** - Modern design with examples  

📈 Training Details

- **Framework:** TensorFlow with Keras API
- **Loss Function:** BinaryCrossentropy (for multi-label classification)
- **Optimizer:** Adam (adaptive learning rate)
- **Activation Functions:**
  - Embedding: None
  - LSTM: tanh
  - Dense layers: ReLU
  - Output: Sigmoid (for probability scores 0-1)
- **Batch Size:** 16 comments per batch
- **Total Training Time:** ~6 hours (3 epochs on CPU/GPU)

📊 Results & Examples

What the Model Does Well ✅

- **Toxic Language Detection:** 99.85% confidence on obvious toxic comments
- **Profanity Detection:** 98.14% confidence on obscene language
- **Insult Detection:** 92.24% confidence on personal attacks

What Could Be Improved ❌

- **Threat Detection:** Sometimes misses subtle threats (13.70% on test example)
- **Identity Hate:** Struggles with coded or subtle identity attacks (30.98%)
- **Severe Toxicity:** Sometimes confuses with regular toxicity

🔄 Data Processing Pipeline
```
Raw Comments (CSV)
        ↓
Load with Pandas
        ↓
Extract text and labels
        ↓
TextVectorization (convert words to numbers)
        ↓
Create TensorFlow Dataset
        ↓
Shuffle (randomize order)
        ↓
Batch (group 16 at a time)
        ↓
Cache (store in memory)
        ↓
Prefetch (prepare next batches)
        ↓
Split (70% train, 20% val, 10% test)
```

🎓 How It Works

1. **Text Preprocessing:** Comments converted to numerical sequences (1800 numbers each)
2. **Embedding:** Words mapped to 32-dimensional meaningful vectors
3. **LSTM Processing:** Bidirectional reading of text for context understanding
4. **Feature Learning:** Dense layers learn complex toxicity patterns
5. **Prediction:** Output layer generates 6 confidence scores (0-1)
6. **Classification:** If score > 0.5, marked as that toxicity type

🚧 Future Improvements

- [ ] Use BERT or Transformers for better accuracy
- [ ] Implement multilingual support
- [ ] Deploy as REST API
- [ ] Add real-time moderation dashboard
- [ ] Support for video/audio comments
- [ ] User feedback loop for continuous improvement
- [ ] Reduce false positives for edge cases

📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Guide](https://keras.io/)
- [Gradio Documentation](https://www.gradio.app/)
- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

👨‍💻 Author

-Chintamani Dinesh Naroji
- College: Late G. N. Sapkal College of Engineering, Nashik
- Internship Project: AI/ML

📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

📧 Contact

For questions or feedback, please reach out through GitHub Issues.

🙏 Acknowledgments

- Jigsaw and Kaggle for the dataset
- TensorFlow/Keras team for excellent documentation
- Everyone who contributed to open-source ML libraries
