from predict_number import models, schemas

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from training_numbers.service import MacrosNN


def get_logs(
    db: Session, offset: int = 0, limit: int = 100
) -> list[schemas.PredictNumber]:
    return (
        db.query(models.PredictNumber)
        .order_by(models.PredictNumber.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def create_number_prediction_log(
    db: Session, prediction_obj: models.PredictNumber, predict: schemas.PredictNumberIn
) -> models.PredictNumber:
    try:
        user_prediction = models.PredictNumber(
            prediction=prediction_obj[MacrosNN.PREDICTED_DIGIT_KEY],
            confidence=prediction_obj[MacrosNN.CONFIDENCE_KEY],
            label=predict.label,
        )
        db.add(user_prediction)
        db.commit()

        return user_prediction
    except SQLAlchemyError as e:
        print(e)
        return None
