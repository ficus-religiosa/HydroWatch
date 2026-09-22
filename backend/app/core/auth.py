from fastapi import Request


async def get_current_client(request: Request) -> str:
    """No-op auth hook; replace with client authentication before production."""
    return "anonymous"