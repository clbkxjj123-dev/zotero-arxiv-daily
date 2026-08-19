from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    title_zh: Optional[str] = None  # title translated into llm.language (None when language is English)
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        translate_title = lang.lower() != 'english'
        if translate_title:
            prompt = (
                "Given the following information of a paper, return a JSON object with exactly two string fields:\n"
                f'  "title_translated": a faithful, fluent translation of the paper title into {lang} '
                "(keep well-known model/method names such as Mamba, U-Net, Transformer in their original form);\n"
                f'  "tldr": a concise summary in {lang} of 3-5 sentences covering: the research problem, '
                "the method or approach, and the key results (include concrete numbers such as accuracy "
                "or improvements when present).\n"
                "Output ONLY the JSON object, no code fences, no preamble. "
                "If only a title or search keywords are provided without a real abstract or full text, "
                'do NOT invent findings; set "tldr" to one short sentence prefixed with "(仅含标题)" '
                "stating what the paper appears to be about based on its title.\n\n"
            )
        else:
            prompt = (
                f"Given the following information of a paper, write a concise summary in {lang} "
                "of 3-5 sentences covering: the research problem, the method or approach, and the "
                "key results (include concrete numbers such as accuracy or improvements when present). "
                "Output only the summary text itself, without any preamble like 'Here is the summary'. "
                "If only a title or search keywords are provided without a real abstract or full text, "
                "do NOT invent findings; instead reply with one short sentence stating what the paper "
                "appears to be about based on its title, prefixed with '(title only)'.\n\n"
            )
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        # An abstract identical to the title is a retriever fallback for
        # abstract-less sources; presenting it as a real abstract would
        # invite the model to fabricate findings.
        if self.abstract and self.abstract != self.title:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        content = response.choices[0].message.content
        if not translate_title:
            return content
        # Parse the {"title_translated", "tldr"} JSON; fall back gracefully so a
        # malformed response degrades to an untranslated title, never a crash.
        try:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            data = json.loads(match.group(0))
            translated = str(data.get("title_translated") or "").strip()
            if translated and translated.lower() != self.title.lower():
                self.title_zh = translated
            return str(data.get("tldr") or "").strip() or content
        except Exception as e:
            logger.warning(f"Failed to parse translated-title JSON for {self.url}: {e}")
            return content

    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]