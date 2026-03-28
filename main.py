import asyncio
import copy
import json
import re
import time

from astrbot.api import logger, sp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register


@register(
    "astrbot_plugin_context_undo",
    "lumingya",
    "按编号回滚当前 AI 对话上下文",
    "2.1.1",
)
class ContextUndoPlugin(Star):
    _MAX_TURNS_PER_UMO = 48
    _STACK_KEY_PREFIX = "undo_turn_stack::"
    _LTM_STATE_KEY_PREFIX = "ltm_state::"
    _PENDING_TURN_EXTRA = "_context_undo_pending_turn"
    _SKIP_LTM_SYNC_EXTRA = "_context_undo_skip_ltm_sync"
    _PREVIEW_LIMIT = 72
    _LIST_LIMIT = 8
    _UNDO_REGEX_PATTERN = (
        r"^(?:撤回|回滚)"
        r"(?:\s*(?:列表|list|ls|(?:最近|last|latest)\s*\d+|第\s*\d+\s*(?:条|轮)|#?\s*\d+\s*(?:号|条|轮)?))?"
        r"[。？！!?]?$"
    )
    _UNDO_COMMAND_NAMES = {"undoctx", "撤回上下文", "回滚上下文"}
    _CLEAR_STACK_COMMAND_PATTERN = re.compile(
        r"(?P<command>/?(?:reset|del|delete))\s*[。？！!?]?",
        re.IGNORECASE,
    )

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self._conversation_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _now_ts() -> int:
        return int(time.time())

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return default

    def _stack_key(self, umo: str, conversation_id: str) -> str:
        return f"{self._STACK_KEY_PREFIX}{umo}::{conversation_id}"

    def _legacy_stack_key(self, umo: str) -> str:
        return f"{self._STACK_KEY_PREFIX}{umo}"

    def _ltm_state_key(self, umo: str, conversation_id: str) -> str:
        return f"{self._LTM_STATE_KEY_PREFIX}{umo}::{conversation_id}"

    def _get_conversation_lock(
        self,
        umo: str,
        conversation_id: str,
    ) -> asyncio.Lock:
        key = f"{umo}::{conversation_id}"
        lock = self._conversation_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[key] = lock
        return lock

    @classmethod
    def _match_clear_stack_command(cls, message_text: str) -> str | None:
        match = cls._CLEAR_STACK_COMMAND_PATTERN.fullmatch(
            (message_text or "").strip()
        )
        if not match:
            return None
        return match.group("command").lstrip("/").lower()

    @staticmethod
    def _coerce_history(raw_history) -> list[dict]:
        if raw_history is None:
            return []

        if isinstance(raw_history, str):
            try:
                raw_history = json.loads(raw_history)
            except json.JSONDecodeError:
                return []

        if not isinstance(raw_history, list):
            return []

        return [copy.deepcopy(item) for item in raw_history if isinstance(item, dict)]

    @staticmethod
    def _coerce_string_list(raw_records) -> list[str]:
        if not isinstance(raw_records, list):
            return []
        return [str(item) for item in raw_records if isinstance(item, str)]

    @staticmethod
    def _normalize_whitespace(text: str | None) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    @classmethod
    def _clip_preview(cls, text: str | None, limit: int | None = None) -> str:
        normalized = cls._normalize_whitespace(text)
        if not normalized:
            return "（空）"

        limit = limit or cls._PREVIEW_LIMIT
        if len(normalized) <= limit:
            return normalized

        return normalized[: max(1, limit - 3)] + "..."

    def _clean_turn_stack(self, raw_stack, now_ts: int | None = None) -> list[dict]:
        if not isinstance(raw_stack, list):
            return []

        now_ts = now_ts or self._now_ts()
        cleaned_stack: list[dict] = []
        for item in raw_stack:
            if not isinstance(item, dict):
                continue

            turn_id = self._safe_int(item.get("turn_id", 0), 0)
            conversation_id = str(item.get("conversation_id", "") or "").strip()
            created_at = self._safe_int(item.get("created_at", 0), 0)
            if turn_id <= 0 or not conversation_id:
                continue

            entry = {
                "turn_id": turn_id,
                "conversation_id": conversation_id,
                "history_before": self._coerce_history(item.get("history_before")),
                "ltm_records_before": self._coerce_string_list(
                    item.get("ltm_records_before")
                ),
                "ltm_records_after": self._coerce_string_list(
                    item.get("ltm_records_after")
                ),
                "user_text": self._clip_preview(item.get("user_text")),
                "assistant_text": self._clip_preview(item.get("assistant_text")),
                "created_at": created_at or now_ts,
            }

            if cleaned_stack and cleaned_stack[-1]["conversation_id"] != conversation_id:
                cleaned_stack = []

            if cleaned_stack and turn_id <= cleaned_stack[-1]["turn_id"]:
                cleaned_stack = [
                    row for row in cleaned_stack if row["turn_id"] < turn_id
                ]

            cleaned_stack.append(entry)

        return cleaned_stack

    async def _load_turn_stack(self, umo: str, conversation_id: str) -> list[dict]:
        key = self._stack_key(umo, conversation_id)
        raw_stack = await self.get_kv_data(key, None)
        if raw_stack is None:
            legacy_key = self._legacy_stack_key(umo)
            legacy_stack = await self.get_kv_data(legacy_key, None)
            cleaned_legacy = self._clean_turn_stack(legacy_stack)
            if cleaned_legacy and all(
                item.get("conversation_id") == conversation_id
                for item in cleaned_legacy
            ):
                await self._save_turn_stack(umo, conversation_id, cleaned_legacy)
                await self.delete_kv_data(legacy_key)
                return cleaned_legacy
            raw_stack = []

        cleaned_stack = self._clean_turn_stack(raw_stack)
        if not isinstance(raw_stack, list):
            await self._save_turn_stack(umo, conversation_id, cleaned_stack)
        elif len(cleaned_stack) != len(raw_stack):
            await self._save_turn_stack(umo, conversation_id, cleaned_stack)
        return cleaned_stack

    async def _save_turn_stack(
        self,
        umo: str,
        conversation_id: str,
        stack: list[dict],
    ) -> None:
        stack = stack[-self._MAX_TURNS_PER_UMO :]
        if stack:
            await self.put_kv_data(self._stack_key(umo, conversation_id), stack)
        else:
            await self.delete_kv_data(self._stack_key(umo, conversation_id))

    async def _clear_turn_stack(
        self,
        umo: str,
        conversation_id: str,
        reason: str,
    ) -> None:
        stack = await self._load_turn_stack(umo, conversation_id)
        if not stack:
            return

        await self._save_turn_stack(umo, conversation_id, [])
        logger.info(
            "Context undo turn stack cleared for %s: cid=%s reason=%s removed=%s",
            umo,
            conversation_id,
            reason,
            len(stack),
        )

    async def _load_ltm_state(
        self,
        umo: str,
        conversation_id: str,
    ) -> list[str] | None:
        key = self._ltm_state_key(umo, conversation_id)
        raw_records = await self.get_kv_data(key, None)
        if raw_records is None:
            return None

        cleaned_records = self._coerce_string_list(raw_records)
        if cleaned_records != raw_records:
            await self._save_ltm_state(umo, conversation_id, cleaned_records)
        return cleaned_records

    async def _save_ltm_state(
        self,
        umo: str,
        conversation_id: str,
        records: list[str],
    ) -> None:
        cleaned_records = self._coerce_string_list(records)
        if cleaned_records:
            await self.put_kv_data(
                self._ltm_state_key(umo, conversation_id),
                cleaned_records,
            )
        else:
            await self.delete_kv_data(self._ltm_state_key(umo, conversation_id))

    def _get_current_ltm_records(self, event: AstrMessageEvent) -> list[str] | None:
        session_chats = self._get_ltm_session_chats()
        if session_chats is None:
            return None

        return [str(item) for item in session_chats.get(event.unified_msg_origin, [])]

    async def _resolve_conversation_ltm_records(
        self,
        umo: str,
        conversation_id: str,
    ) -> list[str]:
        persisted_records = await self._load_ltm_state(umo, conversation_id)
        if persisted_records is not None:
            return persisted_records

        stack = await self._load_turn_stack(umo, conversation_id)
        if not stack:
            return []

        latest_entry = stack[-1]
        latest_records = self._coerce_string_list(latest_entry.get("ltm_records_after"))
        if latest_records:
            return latest_records

        return self._coerce_string_list(latest_entry.get("ltm_records_before"))

    async def _sync_conversation_ltm_state(
        self,
        event: AstrMessageEvent,
        conversation_id: str,
    ) -> list[str] | None:
        session_chats = self._get_ltm_session_chats()
        if session_chats is None or not conversation_id:
            return None

        umo = event.unified_msg_origin
        current_message = event.get_extra("_ltm_current_message")
        current_message = str(current_message) if current_message else ""

        desired_records = list(
            await self._resolve_conversation_ltm_records(umo, conversation_id)
        )
        if current_message and (
            not desired_records or desired_records[-1] != current_message
        ):
            desired_records.append(current_message)

        if desired_records:
            session_chats[umo] = list(desired_records)
        else:
            session_chats.pop(umo, None)

        await self._save_ltm_state(umo, conversation_id, desired_records)
        return desired_records

    async def _clear_ltm_state(
        self,
        event: AstrMessageEvent,
        conversation_id: str,
        reason: str,
    ) -> None:
        umo = event.unified_msg_origin
        cached_records = await self._load_ltm_state(umo, conversation_id)
        current_records = self._get_current_ltm_records(event) or []
        session_chats = self._get_ltm_session_chats()
        if session_chats is not None:
            session_chats.pop(umo, None)

        await self._save_ltm_state(umo, conversation_id, [])
        if cached_records or current_records:
            logger.info(
                "Context undo LTM state cleared for %s: cid=%s reason=%s removed=%s",
                umo,
                conversation_id,
                reason,
                len(cached_records or current_records),
            )

    async def _get_current_turn_stack(
        self,
        umo: str,
        conversation_id: str,
    ) -> list[dict]:
        return await self._load_turn_stack(umo, conversation_id)

    def _get_ltm_session_chats(self):
        star_metadata = self.context.get_registered_star("astrbot")
        star_instance = star_metadata.star_cls if star_metadata else None
        ltm = getattr(star_instance, "ltm", None)
        return getattr(ltm, "session_chats", None)

    def _get_ltm_records_before_turn(self, event: AstrMessageEvent) -> list[str] | None:
        session_chats = self._get_ltm_session_chats()
        if session_chats is None:
            return None

        records = session_chats.get(event.unified_msg_origin, [])
        snapshot = list(records)

        current_ltm_message = event.get_extra("_ltm_current_message")
        if current_ltm_message and snapshot and snapshot[-1] == current_ltm_message:
            snapshot.pop()

        return [str(item) for item in snapshot]

    def _restore_ltm_records(
        self,
        event: AstrMessageEvent,
        records: list[str],
    ) -> None:
        session_chats = self._get_ltm_session_chats()
        if session_chats is None:
            return

        if records:
            session_chats[event.unified_msg_origin] = list(records)
        else:
            session_chats.pop(event.unified_msg_origin, None)

    async def _clear_related_session_state(self, event: AstrMessageEvent) -> None:
        umo = event.unified_msg_origin

        try:
            await sp.session_remove(umo, "session_variables")
        except Exception as exc:
            logger.warning(
                "Context undo failed to clear session_variables for %s: %s",
                umo,
                exc,
            )

        worldbook_meta = self.context.get_registered_star("astrbot_plugin_worldbook")
        worldbook = worldbook_meta.star_cls if worldbook_meta else None
        sessions = getattr(worldbook, "sessions", None)
        clear = getattr(sessions, "clear", None)
        if callable(clear):
            try:
                clear(umo)
            except Exception as exc:
                logger.warning(
                    "Context undo failed to clear worldbook session for %s: %s",
                    umo,
                    exc,
                )

    def _build_user_preview(self, event: AstrMessageEvent, req: ProviderRequest) -> str:
        text = req.prompt or event.message_str or ""
        if self._normalize_whitespace(text):
            return self._clip_preview(text)

        if req.image_urls or req.extra_user_content_parts:
            return "[多模态消息]"

        return "（空消息）"

    def _build_assistant_preview(self, resp: LLMResponse) -> str:
        text = self._normalize_whitespace(resp.completion_text)
        if text:
            return self._clip_preview(text)

        tool_names = self._coerce_tool_names(getattr(resp, "tools_call_name", None))
        if tool_names:
            return self._clip_preview("[工具调用] " + ", ".join(tool_names))

        if resp.result_chain:
            return "[非文本回复]"

        if resp.tools_call_args:
            return "[工具调用]"

        return "[空响应]"

    @staticmethod
    def _coerce_tool_names(raw_names) -> list[str]:
        if isinstance(raw_names, str):
            name = raw_names.strip()
            return [name] if name else []

        if isinstance(raw_names, (list, tuple, set)):
            names: list[str] = []
            for item in raw_names:
                name = str(item).strip()
                if name:
                    names.append(name)
            return names

        return []

    async def _build_pending_turn(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> dict | None:
        conversation = getattr(req, "conversation", None)
        conversation_id = str(getattr(conversation, "cid", "") or "").strip()
        if not conversation_id:
            return None

        async with self._get_conversation_lock(
            event.unified_msg_origin,
            conversation_id,
        ):
            await self._sync_conversation_ltm_state(event, conversation_id)
            history_before = self._coerce_history(getattr(conversation, "history", None))
            if not history_before:
                await self._save_turn_stack(
                    event.unified_msg_origin,
                    conversation_id,
                    [],
                )
            ltm_records = self._get_ltm_records_before_turn(event)

        pending_turn = {
            "conversation_id": conversation_id,
            "history_before": history_before,
            "user_text": self._build_user_preview(event, req),
            "assistant_text": "",
            "created_at": self._now_ts(),
        }

        if ltm_records is not None:
            pending_turn["ltm_records_before"] = ltm_records

        return pending_turn

    @staticmethod
    def _response_is_undoable(resp: LLMResponse) -> bool:
        return bool(
            resp
            and resp.role == "assistant"
            and (resp.completion_text or resp.result_chain or resp.tools_call_args)
        )

    def _extract_command_payload(self, message_text: str) -> str:
        text = (message_text or "").strip()
        if not text:
            return ""

        raw_text = text[1:].strip() if text.startswith("/") else text
        parts = raw_text.split(maxsplit=1)
        head = parts[0].lower() if parts else ""
        if head in self._UNDO_COMMAND_NAMES:
            return parts[1].strip() if len(parts) == 2 else ""

        return raw_text

    def _parse_undo_args(self, raw_args: str) -> tuple[str, int | None]:
        args = self._normalize_whitespace(raw_args)
        if not args:
            return "latest", 1

        lowered = args.lower()
        if lowered in {"列表", "list", "ls"}:
            return "list", None

        match = re.fullmatch(r"(?:最近|last|latest)\s*(\d+)", lowered)
        if match:
            return "latest_count", int(match.group(1))

        match = re.fullmatch(r"第\s*(\d+)\s*(?:条|轮)", args)
        if match:
            return "turn_id", int(match.group(1))

        match = re.fullmatch(r"#?\s*(\d+)\s*(?:号|条|轮)?", args)
        if match:
            return "turn_id", int(match.group(1))

        return "invalid", None

    def _build_usage_text(self) -> str:
        return (
            "用法：\n"
            "1. 撤回 或 回滚：回滚最新一条已编号上下文\n"
            "2. 撤回列表：查看当前对话的编号\n"
            "3. 撤回 #3：回滚到 #3 之前（会移除 #3 及之后的内容）\n"
            "4. 撤回 最近 2：回滚最近 2 条已编号上下文\n"
            "5. /undoctx list | /undoctx 3 | /undoctx last 2"
        )

    @classmethod
    def _match_undo_message(cls, message_text: str):
        return re.fullmatch(cls._UNDO_REGEX_PATTERN, (message_text or "").strip())

    def _format_turn_line(self, entry: dict) -> str:
        return (
            f"#{entry['turn_id']} "
            f"user: {entry['user_text']} | "
            f"assistant: {entry['assistant_text']}"
        )

    def _format_turn_list_text(self, stack: list[dict]) -> str:
        recent_entries = stack[-self._LIST_LIMIT :]
        lines = ["当前对话可撤回的已编号上下文："]
        for entry in recent_entries:
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry["created_at"]))
            lines.append(f"{self._format_turn_line(entry)} [{timestamp}]")
        lines.append("发送“撤回 #编号”可回滚到对应编号之前。")
        lines.append("发送“撤回 最近 2”可一次回滚最近 2 条。")
        return "\n".join(lines)

    def _format_rollback_result(
        self,
        removed_entries: list[dict],
        target_turn_id: int,
    ) -> str:
        first_id = removed_entries[0]["turn_id"]
        last_id = removed_entries[-1]["turn_id"]
        if first_id == last_id:
            header = f"已回滚到 #{target_turn_id} 之前，移除 1 条已编号上下文。"
        else:
            header = (
                f"已回滚到 #{target_turn_id} 之前，"
                f"移除 #{first_id}-#{last_id} 共 {len(removed_entries)} 条已编号上下文。"
            )

        lines = [header]
        preview_entries = removed_entries[:5]
        for index, entry in enumerate(preview_entries, start=1):
            lines.append(f"{index}. {self._format_turn_line(entry)}")

        if len(removed_entries) > len(preview_entries):
            lines.append(f"……其余 {len(removed_entries) - len(preview_entries)} 条已省略。")

        remaining_last_id = target_turn_id - 1
        if remaining_last_id > 0:
            lines.append(f"当前剩余可撤回编号范围：#1-#{remaining_last_id}")
        else:
            lines.append("当前对话已没有剩余的已编号上下文。")

        return "\n".join(lines)

    async def _list_turns(self, event: AstrMessageEvent) -> str:
        umo = event.unified_msg_origin
        conv_mgr = self.context.conversation_manager
        conversation_id = await conv_mgr.get_curr_conversation_id(umo)
        if not conversation_id:
            return "当前没有可撤回的会话。"

        async with self._get_conversation_lock(umo, conversation_id):
            stack = await self._get_current_turn_stack(umo, conversation_id)
        if not stack:
            return "当前对话没有可撤回的已编号上下文。"

        return self._format_turn_list_text(stack)

    async def _rollback_from_stack(
        self,
        event: AstrMessageEvent,
        conv_mgr,
        umo: str,
        conversation_id: str,
        stack: list[dict],
        turn_id: int,
    ) -> tuple[list[dict], int] | str:
        conversation = await conv_mgr.get_conversation(umo, conversation_id)
        if not conversation:
            return "当前会话不存在，无法撤回。"

        if not stack:
            return "当前对话没有可撤回的已编号上下文。"

        target_entry = next(
            (entry for entry in stack if entry["turn_id"] == turn_id),
            None,
        )
        if target_entry is None:
            return f"未找到编号 #{turn_id}。先发送“撤回列表”查看当前可用编号。"

        removed_entries = [entry for entry in stack if entry["turn_id"] >= turn_id]

        await conv_mgr.update_conversation(
            unified_msg_origin=umo,
            conversation_id=conversation_id,
            history=target_entry["history_before"],
            token_usage=0,
        )
        restored_ltm_records = self._coerce_string_list(
            target_entry.get("ltm_records_before")
        )
        self._restore_ltm_records(event, restored_ltm_records)
        await self._save_ltm_state(umo, conversation_id, restored_ltm_records)
        await self._clear_related_session_state(event)

        remaining_stack = [entry for entry in stack if entry["turn_id"] < turn_id]
        await self._save_turn_stack(umo, conversation_id, remaining_stack)
        return removed_entries, turn_id

    async def _rollback_to_turn(self, event: AstrMessageEvent, turn_id: int) -> str:
        if turn_id <= 0:
            return "编号必须大于 0。"

        umo = event.unified_msg_origin
        conv_mgr = self.context.conversation_manager
        conversation_id = await conv_mgr.get_curr_conversation_id(umo)
        if not conversation_id:
            return "当前没有可撤回的会话。"

        async with self._get_conversation_lock(umo, conversation_id):
            stack = await self._get_current_turn_stack(umo, conversation_id)
            result = await self._rollback_from_stack(
                event,
                conv_mgr,
                umo,
                conversation_id,
                stack,
                turn_id,
            )
        if isinstance(result, str):
            return result
        removed_entries, target_turn_id = result

        logger.info(
            "Context undo restored snapshot for %s: cid=%s target_turn=%s removed=%s",
            umo,
            conversation_id,
            target_turn_id,
            len(removed_entries),
        )
        return self._format_rollback_result(removed_entries, target_turn_id)

    async def _rollback_latest_count(
        self,
        event: AstrMessageEvent,
        count: int,
    ) -> str:
        if count <= 0:
            return "回滚数量必须大于 0。"

        umo = event.unified_msg_origin
        conv_mgr = self.context.conversation_manager
        conversation_id = await conv_mgr.get_curr_conversation_id(umo)
        if not conversation_id:
            return "当前没有可撤回的会话。"

        async with self._get_conversation_lock(umo, conversation_id):
            stack = await self._get_current_turn_stack(umo, conversation_id)
            if not stack:
                return "当前对话没有可撤回的已编号上下文。"

            actual_count = min(count, len(stack))
            target_turn_id = stack[-actual_count]["turn_id"]
            result = await self._rollback_from_stack(
                event,
                conv_mgr,
                umo,
                conversation_id,
                stack,
                target_turn_id,
            )
        if isinstance(result, str):
            return result
        removed_entries, target_turn_id = result

        logger.info(
            "Context undo restored snapshot for %s: cid=%s target_turn=%s removed=%s",
            umo,
            conversation_id,
            target_turn_id,
            len(removed_entries),
        )
        return self._format_rollback_result(removed_entries, target_turn_id)

    async def _handle_undo_request(
        self,
        event: AstrMessageEvent,
        raw_args: str,
    ) -> str:
        action, value = self._parse_undo_args(raw_args)
        if action == "latest":
            return await self._rollback_latest_count(event, 1)
        if action == "list":
            return await self._list_turns(event)
        if action == "latest_count" and value is not None:
            return await self._rollback_latest_count(event, value)
        if action == "turn_id" and value is not None:
            return await self._rollback_to_turn(event, value)
        return self._build_usage_text()

    async def _send_plain_result(self, event: AstrMessageEvent, result_text: str) -> None:
        # AstrBot 里传 True 表示禁止默认 LLM 链路继续执行。
        event.should_call_llm(True)
        event.set_extra("skip_history_save", True)
        await event.send(event.plain_result(result_text))

    async def _ensure_admin(self, event: AstrMessageEvent) -> bool:
        if event.is_admin():
            return True

        logger.warning(
            "Context undo rejected for non-admin user %s in %s",
            event.get_sender_id(),
            event.unified_msg_origin,
        )
        await self._send_plain_result(event, "[warning] 仅管理员可以使用上下文撤回。")
        return False

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL, priority=1000)
    async def watch_session_management_commands(self, event: AstrMessageEvent):
        clear_command = self._match_clear_stack_command(event.message_str)
        if clear_command:
            event.set_extra(self._SKIP_LTM_SYNC_EXTRA, True)
            conversation_id = await self.context.conversation_manager.get_curr_conversation_id(
                event.unified_msg_origin
            )
            if conversation_id:
                async with self._get_conversation_lock(
                    event.unified_msg_origin,
                    conversation_id,
                ):
                    await self._clear_turn_stack(
                        event.unified_msg_origin,
                        conversation_id,
                        clear_command,
                    )
                    await self._clear_ltm_state(event, conversation_id, clear_command)

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL, priority=-1000)
    async def sync_live_ltm_state(self, event: AstrMessageEvent):
        if event.get_extra(self._SKIP_LTM_SYNC_EXTRA):
            self._restore_ltm_records(event, [])
            return

        conversation_id = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        if not conversation_id:
            return

        async with self._get_conversation_lock(
            event.unified_msg_origin,
            conversation_id,
        ):
            await self._sync_conversation_ltm_state(event, conversation_id)

    @filter.on_llm_request(priority=1000)
    async def record_pending_turn(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        pending_turn = await self._build_pending_turn(event, req)
        if pending_turn:
            event.set_extra(self._PENDING_TURN_EXTRA, pending_turn)

    @filter.on_llm_response(priority=-1000)
    async def commit_turn(
        self,
        event: AstrMessageEvent,
        resp: LLMResponse,
    ):
        pending_turn = event.get_extra(self._PENDING_TURN_EXTRA)
        event.set_extra(self._PENDING_TURN_EXTRA, None)
        if not isinstance(pending_turn, dict):
            return
        if not self._response_is_undoable(resp):
            return

        pending_turn = copy.deepcopy(pending_turn)
        pending_turn["assistant_text"] = self._build_assistant_preview(resp)

        async with self._get_conversation_lock(
            event.unified_msg_origin,
            pending_turn["conversation_id"],
        ):
            ltm_records_after = self._get_current_ltm_records(event)
            if ltm_records_after is not None:
                pending_turn["ltm_records_after"] = ltm_records_after

            stack = await self._get_current_turn_stack(
                event.unified_msg_origin,
                pending_turn["conversation_id"],
            )
            if not pending_turn.get("history_before"):
                stack = []
            pending_turn["turn_id"] = (stack[-1]["turn_id"] + 1) if stack else 1
            stack.append(pending_turn)
            await self._save_turn_stack(
                event.unified_msg_origin,
                pending_turn["conversation_id"],
                stack,
            )
            if ltm_records_after is not None:
                await self._save_ltm_state(
                    event.unified_msg_origin,
                    pending_turn["conversation_id"],
                    ltm_records_after,
                )

        logger.info(
            "Context undo checkpoint saved for %s: cid=%s turn_id=%s stack=%s",
            event.unified_msg_origin,
            pending_turn["conversation_id"],
            pending_turn["turn_id"],
            len(stack),
        )

    @filter.command(
        "undoctx",
        alias={"撤回上下文", "回滚上下文"},
    )
    async def undo_context_command(self, event: AstrMessageEvent):
        if not await self._ensure_admin(event):
            return

        result_text = await self._handle_undo_request(
            event,
            self._extract_command_payload(event.message_str),
        )
        await self._send_plain_result(event, result_text)

    @filter.regex(_UNDO_REGEX_PATTERN)
    async def undo_context_regex(self, event: AstrMessageEvent):
        match = self._match_undo_message(event.message_str)
        if not match:
            return

        if not await self._ensure_admin(event):
            return

        command_text = (event.message_str or "").strip()
        payload = re.sub(r"^(?:撤回|回滚)\s*", "", command_text)
        payload = re.sub(r"[。？！!?]+$", "", payload).strip()
        result_text = await self._handle_undo_request(event, payload)
        await self._send_plain_result(event, result_text)
