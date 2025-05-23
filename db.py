import asyncpg
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

class PlateDatabase:
  def __init__(self):
    load_dotenv()

    self.user = os.getenv("DB_USER")
    self.password = os.getenv("DB_PASSWORD")
    self.database = os.getenv("DB_NAME")
    self.host = os.getenv("DB_HOST")
    self.port = int(os.getenv("DB_PORT", 5432))
    self.pool = None

  async def init(self):
    self.pool = await asyncpg.create_pool(
      user=self.user,
      password=self.password,
      database=self.database,
      host=self.host,
      port=self.port,
      min_size=1,
      max_size=10
    )
    await self._init_schema()

  async def _init_schema(self):
    async with self.pool.acquire() as conn:
      await conn.execute("""
        CREATE TABLE IF NOT EXISTS plates (
          id SERIAL PRIMARY KEY,
          timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          plate TEXT NOT NULL
        );
      """)

  async def insert_plate(self, plate: str, timestamp: datetime):
    async with self.pool.acquire() as conn:
      await conn.execute(
        "INSERT INTO plates (timestamp, plate) VALUES ($1, $2)",
        timestamp, plate
      )

  async def get_all(self):
    async with self.pool.acquire() as conn:
        return await conn.fetch("SELECT * FROM plates ORDER BY timestamp DESC")
  
  async def clear_table(self):
    async with self.pool.acquire() as conn:
      await conn.execute("""
        DROP TABLE IF EXISTS plates;
      """)
    await self._init_schema()
    

  async def close(self):
    if self.pool:
        await self.pool.close()