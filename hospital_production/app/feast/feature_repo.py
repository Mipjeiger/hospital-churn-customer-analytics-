from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64, String
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.value_type import ValueType

patient = Entity(name="patient_id", join_keys=["patient_id"], value_type=ValueType.STRING)

"""
Feature store using Feast that connects to a PostgreSQL database. 
The features are defined in the patient_features FeatureView.
"""

source = PostgreSQLSource(
    name="hospital_data_source",
    query="""
        SELECT
            patient_id,
            arrival_date,
            age,
            satisfaction,
            length_of_stay,
            available_beds,
            patients_request,
            patients_admitted,
            patients_refused,
            staff_morale,
            staff_present,
            total_staff,
            staff_absens AS staff_absent,
            (patients_request / NULLIF(available_beds + 1, 0))::float
                AS capacity_pressure,
            (patients_admitted / NULLIF(patients_request, 0))::float
                AS admission_rate_row,
            (patients_refused / NULLIF(patients_request, 0))::float
                AS refusal_rate_row,
            (staff_present / NULLIF(total_staff, 0))::float
                AS staff_attendance_rate,
            service
        FROM hospital.hospital_data
    """,
    timestamp_field="arrival_date",
)

patient_features = FeatureView(
    name="patient_features",
    entities=[patient],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age",                    dtype=Int64),
        Field(name="satisfaction",           dtype=Int64),
        Field(name="length_of_stay",         dtype=Int64),
        Field(name="available_beds",         dtype=Int64),
        Field(name="patients_request",       dtype=Float32),
        Field(name="patients_admitted",      dtype=Float32),
        Field(name="patients_refused",       dtype=Float32),
        Field(name="staff_morale",           dtype=Float32),
        Field(name="staff_present",          dtype=Float32),
        Field(name="total_staff",            dtype=Int64),
        Field(name="staff_absent",           dtype=Float32),
        Field(name="capacity_pressure",      dtype=Float32),
        Field(name="admission_rate_row",     dtype=Float32),
        Field(name="refusal_rate_row",       dtype=Float32),
        Field(name="staff_attendance_rate",  dtype=Float32),
        Field(name="service",                dtype=String),
    ],
    source=source,
)