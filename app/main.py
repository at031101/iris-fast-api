from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import uvicorn

app = FastAPI(title="Iris Classification API")

# Load and train model once (for demo)
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
joblib.dump(model, "model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris Model API is live!"}

@app.post("/predict")
def predict(features: IrisInput):
    try:
        model = joblib.load("model.joblib")
        data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]
        prediction = model.predict(data)[0]
        return {"predicted_class": iris.target_names[prediction]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
