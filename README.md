# LungPredict - AI-Powered Lung Disease Detection

A sophisticated Flask web application that leverages deep learning to detect respiratory conditions from chest X-ray images. Built with EfficientNetB0 architecture, this application provides accurate predictions with detailed analysis and health recommendations.

## 🌟 Features

- **Advanced AI Analysis** - Uses EfficientNetB0 neural network with 96% accuracy
- **Multi-Disease Detection** - Identifies COVID-19, Viral Pneumonia, and Normal lung conditions
- **Secure & Private** - Health data is protected and never shared with third parties
- **Comprehensive Reports** - Detailed analysis with confidence scores and health recommendations
- **User-Friendly Interface** - Modern, responsive design with smooth animations
- **History Tracking** - Stores predictions for future reference by healthcare providers

## 📸 Application Screenshots

### Home Page
![Home Page](Image%201.png)
*Welcome screen with key features and "Let's Start" button*

### About Page - Mission
![About Page](Image%202.png)
*Information about LungPredict's mission and technology*

### About Page - Technology Stack
![Technology Stack](Image%203.png)
*Technology stack: Flask, TensorFlow, PostgreSQL, and Bootstrap*

### Personal Information Form
![Personal Information](Image%204.png)
*Patient details collection form*

### X-Ray Upload
![X-Ray Upload](Image%205.png)
*Drag-and-drop X-ray image upload with progress indicator*

### Prediction Results
![Prediction Results](Image%206.png)
*Detailed prediction outcome with confidence score*

### Detailed Analysis
![Detailed Analysis](Image%207.png)
*Comprehensive analysis of lung characteristics*

### Health Recommendations
![Health Recommendations](Image%208.png)
*Top 5 precautions and health advice based on prediction*

### Contact Page
![Contact Page](Image%209.png)
*Meet the developer and contact information*

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras with EfficientNetB0
- **Database**: PostgreSQL
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Additional Libraries**: NumPy, Pillow, Werkzeug

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- PostgreSQL database (optional, for history feature)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yashshinde96/LungPredict.git
cd LungPredict
```

### 2. Install Virtual Environment
Install the virtualenv package if you don't have it already:
```bash
pip install virtualenv
```

### 3. Create a Virtual Environment
Create a new virtual environment inside your project folder:
```bash
virtualenv venv
```

### 4. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 5. Install Required Dependencies
Install all necessary Python packages from requirements.txt:
```bash
pip install -r requirements.txt
```

### 6. Set Up the Model
Make sure the pre-trained model file is in the correct location:
```
models/EfficientNetB0-Lung-96.h5
```

### 7. Configure Database (Optional)
If you want to use the history feature, set up your PostgreSQL database and update the connection string in `app.py`.

### 8. Run the Flask App
```bash
python app.py
```

The app will be hosted locally at:
```
http://127.0.0.1:5000
```

Copy the above link and paste it into any browser.

## 📁 Project Structure

```
LungPredict/
├── app.py                  # Main Flask application
├── models/
│   └── EfficientNetB0-Lung-96.h5  # Pre-trained model
├── static/
│   ├── img1.jpg           # Background images
│   ├── MyPic.jpg          # Developer profile picture
│   └── uploads/           # Uploaded X-ray images
├── templates/
│   ├── index.html         # Home page
│   ├── about.html         # About page
│   ├── contact.html       # Contact page
│   ├── login.html         # History/Login page
│   ├── predict.html       # Prediction form
│   └── result.html        # Results page
├── requirements.txt       # Python dependencies
├── Image 1.png           # Screenshot: Home Page
├── Image 2.png           # Screenshot: About Page - Mission
├── Image 3.png           # Screenshot: Technology Stack
├── Image 4.png           # Screenshot: Personal Information Form
├── Image 5.png           # Screenshot: X-Ray Upload
├── Image 6.png           # Screenshot: Prediction Results
├── Image 7.png           # Screenshot: Detailed Analysis
├── Image 8.png           # Screenshot: Health Recommendations
├── Image 9.png           # Screenshot: Contact Page
└── README.md             # This file
```

## 🎯 How to Use

1. **Start Application** - Click "Let's Start" on the home page
2. **Enter Details** - Fill in personal information (name, age, gender)
3. **Upload X-Ray** - Upload a chest X-ray image (JPG, PNG, or JPEG)
4. **View Results** - Get instant prediction with confidence score
5. **Read Analysis** - Review detailed lung characteristics analysis
6. **Follow Recommendations** - Check health precautions and advice
7. **Download Report** - Save PDF report for medical consultation

## 🔒 Privacy & Security

- All health data is encrypted and stored securely
- Images and predictions are only accessible by authorized users
- No data is shared with third parties
- Compliant with healthcare data protection standards

## ⚠️ Disclaimer

**This is an AI-assisted prediction tool. Always consult a healthcare professional for a definitive diagnosis.** The predictions provided by this application are for informational purposes only and should not replace professional medical advice.

## 👨‍💻 Developer

**Yash Shinde**
- Email: yashshinde570@gmail.com
- LinkedIn: [Yash Shinde](https://www.linkedin.com/in/yash-shinde-42853836b)
- GitHub: [yashshinde96](https://github.com/yashshinde96)

## 🏫 Institution

VSM College of Engineering, Nipani

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- EfficientNetB0 architecture by Google Research
- Bootstrap team for the responsive framework
- Medical imaging datasets used for training
- VSM College of Engineering for support

## 🐛 Bug Reports & Feature Requests

If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository or contact the developer directly.

## 📊 Model Performance

- **Accuracy**: 96%
- **Training Dataset**: Verified chest X-ray images
- **Classes**: COVID-19, Viral Pneumonia, Normal
- **Architecture**: EfficientNetB0 with custom top layers

---

**Made with ❤️ for better healthcare accessibility**