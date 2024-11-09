# api.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# model CNN (harus sama dengan struktur model training)
class CovidCNN(nn.Module):
    def __init__(self):
        super(CovidCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

app = Flask(__name__)

# Definisi transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model yang sudah di-training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidCNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Class labels
classes = ['NORMAL', 'PNEUMONIA']

#Membuat endpoint API dengan metode POST 
@app.route('/predict', methods=['POST'])
def predict():
    # Kode ini aktif jika endpoint menerima request tapi tidak menerima file dikategorikan sebagai eror 400
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    #memproses file 
    try:
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, 1)[0]
            _, predicted = torch.max(output.data, 1)
            result = classes[predicted.item()]
            confidence = float(probabilities[predicted.item()].cpu())
        
        # Kode di bawah ini akan diumpankan kembali ke antarmuka postman setelah berhasil dilakukan prediksi
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'probabilities': {
                classes[i]: float(prob) 
                for i, prob in enumerate(probabilities.cpu())
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Model loaded and ready for predictions!") #tulisan ini muncul pada server, bukan di aplikasi postman ya
    app.run(host='0.0.0.0', port=5000, debug=False) 
    # Nilai 0.0.0.0 berarti api dapat diakses di semua device secara global dan tersedia di port 5000. 
    #PORT bisa diubah jika perlu, biasanya karena ada 