import asyncio
import aiohttp

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(5):  # 同时5个请求
            tasks.append(session.post(
                "http://localhost:8000/predict",
                json={"text": f"This is test {i}"}
            ))
        responses = await asyncio.gather(*tasks)
        for r in responses:
            print(await r.json())

asyncio.run(main())
