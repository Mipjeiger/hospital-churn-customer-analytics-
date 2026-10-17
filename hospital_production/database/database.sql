-- Show all schemas in the database
SELECT schema_name
FROM information_schema.schemata;

-- Create a new table for database
CREATE TABLE hospital.hospital_data (
    patient_id VARCHAR(20)  PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age SMALLINT NOT NULL CHECK (age >= 0 AND age <= 120),
    arrival_date TIMESTAMP NOT NULL,
    departure_date       TIMESTAMP    NOT NULL,
    service              VARCHAR(50)  NOT NULL,
    satisfaction         SMALLINT     NOT NULL CHECK (satisfaction BETWEEN 0 AND 100),
    month                SMALLINT     NOT NULL CHECK (month BETWEEN 1 AND 12),
    week                 SMALLINT     NOT NULL CHECK (week BETWEEN 1 AND 53),
    available_beds       INTEGER      NOT NULL CHECK (available_beds >= 0),
    patients_request     DOUBLE PRECISION NOT NULL CHECK (patients_request >= 0),
    patients_admitted    DOUBLE PRECISION NOT NULL CHECK (patients_admitted >= 0),
    patients_refused     DOUBLE PRECISION NOT NULL CHECK (patients_refused >= 0),
    patient_satisfaction DOUBLE PRECISION CHECK (patient_satisfaction BETWEEN 0 AND 100),
    staff_morale         DOUBLE PRECISION CHECK (staff_morale BETWEEN 0 AND 100),
    total_staff          INTEGER      NOT NULL CHECK (total_staff >= 0),
    staff_present        DOUBLE PRECISION NOT NULL CHECK (staff_present >= 0),
    staff_absens         DOUBLE PRECISION NOT NULL CHECK (staff_absens >= 0),
    length_of_stay       INTEGER      NOT NULL CHECK (length_of_stay >= 0),
    patient_churn        SMALLINT     NOT NULL CHECK (patient_churn IN (0, 1))
);

SELECT * FROM hospital.hospital_data;

SELECT SCHEMA_NAME
FROM INFORMATION_SCHEMA.SCHEMATA
WHERE SCHEMA_NAME = 'hospital';