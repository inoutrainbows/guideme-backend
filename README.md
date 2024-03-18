# How to run this

python3 -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt

curl -X POST -F "file=@cat.jpg" http://localhost:5000/predict
