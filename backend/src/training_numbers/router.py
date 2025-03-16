from fastapi import APIRouter

from service import TrainingNumbersNN


router = APIRouter(
    prefix="/training-numbers",
    tags=["training-numbers"],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
async def training_numbers() -> None:
    print("Starting train the number model...")
    tn = TrainingNumbersNN()
    tn.train_numbers()
    return
