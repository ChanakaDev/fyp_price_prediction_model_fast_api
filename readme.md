# This directory is there for the price prediction model

- change directory `cd fast_api_backend/price_prediction_model_api`
- create a venv `python3 -m venv venv`
- activate `source venv/bin/activate`

- install uvicorn `pip3 install fastapi uvicorn`
- run the project `uvicorn main:app --reload`

- check pip packages in that venv `pip3 list`
- installing packages for that venv `pip3 install <package_1> <package_2>`
- create a requirements file `pip3 freeze > requirements.txt`

- deactivate `deactivate`