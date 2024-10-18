open terminal in the same dir and run this

Activate Virtual enviournment first
create:
	python3 -m venv env
Activate:
	.\env\Scripts\activate

pip3 install -r requirements.txt

python -m flask --app app.py run
