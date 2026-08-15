def main() -> None:
    """CLI 入口：启动 MCP streamable HTTP 服务器。"""
    from dsv_mcp.server import main as _server_main

    _server_main()
