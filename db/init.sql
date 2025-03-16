CREATE TABLE predict_number (
    id SERIAL PRIMARY KEY,
    timestamp NOT NULL TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    label INT NOT NULL,
    prediction INT NOT NULL,
    confidence REAL NOT NULL
);

INSERT INTO predict_number (prediction, label, confidence, timestamp) VALUES (3, 6, 97.6, NOW()), (3, 8, 86.1, NOW());