from typing import Annotated

from predict_number.dependencies import get_db
from predict_number import schemas
from predict_number.service import PredictNumber
from predict_number.crud import get_logs, create_number_prediction_log

from fastapi import APIRouter, Depends, HTTPException, Query


router = APIRouter(
    prefix="/predict-number",
    tags=["predict-number"],
    responses={404: {"description": "Not found"}},
)


@router.get("/read/logs", response_model=list[schemas.PredictNumber])
async def read_logs(
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
    db=Depends(get_db),
):
    result = get_logs(db=db, offset=offset, limit=limit)
    if not result:
        raise HTTPException(status_code=404, detail="There is no logs!")
    return result


@router.post("/read/number", response_model=schemas.PredictNumber)
async def read_number(
    predict: schemas.PredictNumberIn,
    db=Depends(get_db),
):
    predict_number_obj = PredictNumber()
    prediction_obj = predict_number_obj.predict_digit(predict.b64_encoded)
    if not prediction_obj:
        raise HTTPException(status_code=404, detail="Failed to predict image number!")
    # Persists (save in db) the information about the user and prediction.
    user_prediction = create_number_prediction_log(
        db=db, predict=predict, prediction_obj=prediction_obj
    )
    if not user_prediction:
        raise HTTPException(status_code=404, detail="Failed to predict image number!")

    return user_prediction
