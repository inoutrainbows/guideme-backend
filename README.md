# How to run this

python3 -m venv myenv

source myenv/bin/activate

pip3 install -r requirements.txt

python3 app.py

curl -X POST -F "image_data=@cat.jpg" http://localhost:8080/predict
