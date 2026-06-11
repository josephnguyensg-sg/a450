# tool5.py
import logging
import os
from typing import Dict, Optional, Any
import polars as pl
import json as _json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load cấu hình từ tool5.json ───────────────────────────────────────────
with open(os.path.join(_HERE, "tool5.json"), encoding="utf-8") as _f:
    _CFG = _json.load(_f)

DEFAULT_TABLES = _CFG["default_tables"]

class ParquetSQLTool:
    """
    Công cụ để đăng ký file Parquet vào SQLContext của Polars
    và thực thi các truy vấn SQL, trả về DataFrame đã collect().
    Các bảng mặc định sẽ được đăng ký tự động khi khởi tạo.
    """

    def __init__(self, default_tables: Optional[Dict[str, str]] = None):
        self.ctx = pl.SQLContext()
        self._registered = set()
        # Lấy mapping mặc định: tham số > biến môi trường (JSON) > DEFAULT_TABLES
        env_mapping = os.getenv("TOOL5_DEFAULT_TABLES")
        if default_tables is None and env_mapping:
            try:
                import json
                default_tables = json.loads(env_mapping)
            except Exception:
                default_tables = None
        self.default_tables = default_tables or DEFAULT_TABLES
        # Đăng ký các bảng mặc định ngay khi khởi tạo; lỗi sẽ được log nhưng không dừng khởi tạo
        for name, path in self.default_tables.items():
            try:
                self.register_table(name, path, overwrite=False)
            except Exception:
                logger.warning("Auto-register failed for %s -> %s", name, path)

    def _format_error(self, e: Exception) -> str:
        return f"{type(e).__name__}: {str(e)}"

    def register_table(self, name: str, path: str, overwrite: bool = False) -> None:
        """
        Đăng ký một file Parquet (scan_parquet) dưới tên bảng SQL.
        name: tên bảng trong SQLContext
        path: đường dẫn tới file parquet
        overwrite: nếu True và bảng đã tồn tại thì ghi đè
        """
        if name in self._registered and not overwrite:
            logger.info("Table %s already registered; skip (set overwrite=True to re-register).", name)
            return
        try:
            self.ctx.register(name, pl.scan_parquet(path))
            self._registered.add(name)
            logger.info("Registered table %s -> %s", name, path)
        except Exception as e:
            logger.exception("Failed to register table %s from %s: %s", name, path, e)
            raise

    def register_table_safe(self, name: str, path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Phiên bản an toàn: không raise, trả về dict {'success': bool, 'error': str|None}
        """
        try:
            self.register_table(name, path, overwrite=overwrite)
            return {"success": True, "error": None}
        except Exception as e:
            err = self._format_error(e)
            return {"success": False, "error": err}

    def register_tables(self, mapping: Dict[str, str], overwrite: bool = False) -> None:
        """
        Đăng ký nhiều bảng cùng lúc.
        mapping: dict {table_name: parquet_path}
        """
        for name, path in mapping.items():
            self.register_table(name, path, overwrite=overwrite)

    def register_tables_safe(self, mapping: Dict[str, str], overwrite: bool = False) -> Dict[str, Any]:
        """
        Phiên bản an toàn cho register_tables: trả về tổng hợp kết quả cho từng bảng.
        Trả về: {"success": bool, "details": {table: {"success": bool, "error": str|None}}}
        """
        details: Dict[str, Dict[str, Optional[str]]] = {}
        overall_success = True
        for name, path in mapping.items():
            res = self.register_table_safe(name, path, overwrite=overwrite)
            details[name] = {"success": res["success"], "error": res["error"]}
            if not res["success"]:
                overall_success = False
        return {"success": overall_success, "details": details}

    def execute_sql(self, query: str, collect: bool = True):
        """
        Thực thi câu lệnh SQL trên SQLContext đã đăng ký.
        Nếu collect=True thì trả về DataFrame đã collect(); nếu False trả về LazyFrame.
        """
        result = self.ctx.execute(query)
        if collect:
            df = result.collect()
            logger.info("Query executed and collected. Rows: %s", len(df))
            return df
        else:
            return result

    def execute_sql_safe(self, query: str, collect: bool = True) -> Dict[str, Any]:
        """
        Phiên bản an toàn: không raise, trả về dict:
        {
            "success": bool,
            "data": polars.DataFrame | None,
            "rows": int | None,
            "error": str | None
        }
        """
        try:
            result = self.ctx.execute(query)
            if collect:
                df = result.collect()
                rows = len(df)
                logger.info("Query executed and collected. Rows: %s", rows)
                return {"success": True, "data": df, "rows": rows, "error": None}
            else:
                return {"success": True, "data": result, "rows": None, "error": None}
        except Exception as e:
            err = self._format_error(e)
            logger.exception("SQL execution failed: %s", e)
            return {"success": False, "data": None, "rows": None, "error": err}

    def execute_and_save(self, query: str, out_path: str) -> None:
        """
        Thực thi query, collect kết quả và lưu ra parquet.
        """
        df = self.execute_sql(query, collect=True)
        try:
            df.write_parquet(out_path)
            logger.info("Saved query result to %s", out_path)
        except Exception as e:
            logger.exception("Failed to save result to %s: %s", out_path, e)
            raise

    def execute_and_save_safe(self, query: str, out_path: str) -> Dict[str, Any]:
        """
        Phiên bản an toàn cho execute_and_save: trả về dict {"success": bool, "error": str|None}
        """
        try:
            res = self.execute_sql_safe(query, collect=True)
            if not res["success"]:
                return {"success": False, "error": res["error"]}
            df = res["data"]
            try:
                df.write_parquet(out_path)
                logger.info("Saved query result to %s", out_path)
                return {"success": True, "error": None}
            except Exception as e:
                err = self._format_error(e)
                logger.exception("Failed to save result to %s: %s", out_path, e)
                return {"success": False, "error": err}
        except Exception as e:
            err = self._format_error(e)
            logger.exception("Unexpected failure in execute_and_save_safe: %s", e)
            return {"success": False, "error": err}

# Singleton helper để agent gọi nhanh
_tool_instance: Optional[ParquetSQLTool] = None

def get_tool(default_tables: Optional[Dict[str, str]] = None) -> ParquetSQLTool:
    """
    Lấy instance singleton.
    - default_tables có thể truyền mapping riêng để ghi đè DEFAULT_TABLES.
    Các bảng mặc định sẽ luôn được đăng ký khi instance được tạo.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = ParquetSQLTool(default_tables=default_tables)
    return _tool_instance
