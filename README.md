# Environment
rename .env.example thành .env
API key trong .env không phải thật, nếu cần API key để access vui lòng liên hệ Huy

# Requirements
run:
pip install -r requirement.txt 

# Tasks:
- Fetch data from API to save in SQLite: Huy
- Flask Back End:
    + List: Huy
    + Detail: Huy
    + Search: Linh
    + Compare options: Linh
- UI Front End:
    + List page (+ search): Long, Linh, Huy
    + Detail page: Long, Linh, Huy
- PowerPoint reports:
    + Introduction: Long
    + Scope: Long, Huy
    + Workflow: Phú
    + Technology: Phú
    + API: Huy
    + Product: Phú
- Reporters: Phú, Linh

# Powerpoint report:
URL: https://docs.google.com/presentation/d/1xEjVgKdxs9JWmNdzEtORnodRCYbiF7EBlV0BompwTB8/edit?usp=sharing

# Errors:
- [NoneType] object: Vào .env, đổi biến AVIATION_DATA_SAVED='false', và run python data_retriever.py
