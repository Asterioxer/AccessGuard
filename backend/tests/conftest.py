import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure test settings
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.db.session import Base, get_db
from app.db.models import User, PedestrianSegment
from app.core.security import get_password_hash, create_access_token
from app.gis.osm_importer import create_pedestrian_segment_models
from app.main import app

# In-memory test SQLite DB
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    # Seed default segments for tests
    segs = create_pedestrian_segment_models()
    for s in segs:
        session.add(s)
    session.commit()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def contributor_user(db_session):
    user = User(
        email="contributor@test.org",
        hashed_password=get_password_hash("testpass123"),
        role="contributor",
        reputation_score=15
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def verifier_user(db_session):
    user = User(
        email="verifier@test.org",
        hashed_password=get_password_hash("testpass123"),
        role="verifier",
        reputation_score=60
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def admin_user(db_session):
    user = User(
        email="admin@test.org",
        hashed_password=get_password_hash("testpass123"),
        role="admin",
        reputation_score=100
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def contributor_token(contributor_user):
    return create_access_token(data={"sub": contributor_user.id, "email": contributor_user.email, "role": contributor_user.role})

@pytest.fixture
def verifier_token(verifier_user):
    return create_access_token(data={"sub": verifier_user.id, "email": verifier_user.email, "role": verifier_user.role})

@pytest.fixture
def admin_token(admin_user):
    return create_access_token(data={"sub": admin_user.id, "email": admin_user.email, "role": admin_user.role})
