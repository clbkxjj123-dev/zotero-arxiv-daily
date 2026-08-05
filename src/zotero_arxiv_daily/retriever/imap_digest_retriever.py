import email
import imaplib
import re
from email.message import Message
from typing import Any
from urllib.parse import unquote

from loguru import logger

from .base import BaseRetriever, register_retriever
from .html_text import strip_html
from ..protocol import Paper

# One entry in a Google Scholar alert email: linked title, green byline
# (authors - venue, year), then the snippet div.
GS_ENTRY_RE = re.compile(
    r'<a[^>]+class="gse_alrt_title"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r".*?<div[^>]*color:#006621[^>]*>(?P<byline>.*?)</div>"
    r'.*?<div[^>]*class="gse_alrt_sni"[^>]*>(?P<snippet>.*?)</div>',
    re.DOTALL | re.IGNORECASE,
)
# Fallback for other alert providers (CNKI, Wanfang, publisher alerts):
# any link whose visible text is long enough to be a paper title.
GENERIC_LINK_RE = re.compile(
    r'<a[^>]+href="(?P<url>https?://[^"]+)"[^>]*>(?P<title>[^<]{12,})</a>',
    re.IGNORECASE,
)
IGNORED_URL_KEYWORDS = ("unsubscribe", "privacy", "settings", "preferences", "help", "support")


@register_retriever("imap_digest")
class ImapDigestRetriever(BaseRetriever):
    """Parse subscription digest emails (Google Scholar alerts, CNKI/Wanfang
    journal alerts, ...) from an IMAP mailbox into papers.

    Only unread mails from the configured senders are touched; successfully
    parsed mails are marked as read so they are processed exactly once.
    """

    def __init__(self, config):
        super().__init__(config)
        cfg = self.retriever_config
        for field in ("server", "user", "password", "senders"):
            if not cfg.get(field):
                raise ValueError(f"source.imap_digest.{field} must be set")

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        cfg = self.retriever_config
        entries: list[dict[str, Any]] = []
        try:
            mail = imaplib.IMAP4_SSL(cfg.server, cfg.get("port", 993))
            mail.login(cfg.user, cfg.password)
            self._send_imap_id(mail)
            mail.select(cfg.get("folder", "INBOX"))
            processed_uids = []
            max_emails = cfg.get("max_emails", 50)
            for sender in cfg.senders:
                status, data = mail.uid("SEARCH", None, f'(UNSEEN FROM "{sender}")')
                if status != "OK" or not data or not data[0]:
                    continue
                for uid in data[0].split()[-max_emails:]:
                    status, msg_data = mail.uid("FETCH", uid, "(BODY.PEEK[])")
                    if status != "OK" or not msg_data or msg_data[0] is None:
                        continue
                    message = email.message_from_bytes(msg_data[0][1])
                    parsed = self._parse_entries(self._extract_html(message), str(sender))
                    if parsed:
                        entries.extend(parsed)
                        processed_uids.append(uid)
            if cfg.get("mark_seen", True):
                for uid in processed_uids:
                    mail.uid("STORE", uid, "+FLAGS", "(\\Seen)")
            mail.logout()
        except Exception as e:
            # A broken mailbox must not kill the whole daily run
            logger.error(f"Failed to retrieve IMAP digests: {e}")
        seen_titles: set[str] = set()
        unique_entries = []
        for entry in entries:
            key = entry["title"].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                unique_entries.append(entry)
        if self.config.executor.debug:
            unique_entries = unique_entries[:10]
        return unique_entries

    @staticmethod
    def _send_imap_id(mail: imaplib.IMAP4_SSL) -> None:
        """Some Chinese providers (QQ, NetEase) reject clients that do not
        identify themselves via the IMAP ID extension."""
        try:
            imaplib.Commands["ID"] = ("AUTH", "SELECTED")
            mail._simple_command("ID", '("name" "zotero-arxiv-daily" "version" "1.0.0")')
        except Exception as e:
            logger.debug(f"IMAP ID command not accepted (usually harmless): {e}")

    @staticmethod
    def _extract_html(message: Message) -> str:
        parts = message.walk() if message.is_multipart() else [message]
        html, plain = "", ""
        for part in parts:
            content_type = part.get_content_type()
            if content_type not in ("text/html", "text/plain"):
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if content_type == "text/html":
                html += text
            else:
                plain += text
        return html or plain

    def _parse_entries(self, html: str, sender: str) -> list[dict[str, Any]]:
        entries = [
            {
                "title": strip_html(m.group("title")),
                "byline": strip_html(m.group("byline")),
                "snippet": strip_html(m.group("snippet")),
                "url": self._unwrap_redirect(m.group("url")),
                "sender": sender,
            }
            for m in GS_ENTRY_RE.finditer(html)
        ]
        if entries:
            return entries
        for m in GENERIC_LINK_RE.finditer(html):
            url = m.group("url")
            if any(keyword in url.lower() for keyword in IGNORED_URL_KEYWORDS):
                continue
            title = strip_html(m.group("title"))
            if not title:
                continue
            entries.append({"title": title, "byline": "", "snippet": "", "url": url, "sender": sender})
        return entries

    @staticmethod
    def _unwrap_redirect(url: str) -> str:
        """Google Scholar links point at scholar_url?url=<target>&...; unwrap them."""
        if match := re.search(r"[?&]url=([^&]+)", url):
            return unquote(match.group(1))
        return url

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = raw_paper["title"]
        if not title:
            return None
        # Byline looks like "A Author, B Author - Journal, 2026 - publisher"
        authors = []
        if raw_paper["byline"]:
            authors = [a.strip() for a in raw_paper["byline"].split(" - ")[0].split(",")]
            authors = [a for a in authors if a]
        # Fall back to the title so snippet-less entries can still be scored
        abstract = raw_paper["snippet"] or title
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper["url"],
        )
