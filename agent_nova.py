#!/usr/bin/env python3
"""
AGENT IA NOVA — Application complète (fichier unique)
======================================================
Fusion de trois briques en un seul fichier à lancer :
  1. Nova Safe Core   : chiffrement, analyse statique, scaffolding de projet
  2. Agent IA         : logique de l'agent conversationnel (tool use / multi-fournisseurs)
  3. Interface Web     : page de chat stylée, servie par un serveur Flask local

Deux modes d'utilisation du MÊME fichier :

  A) Interface web (par défaut) :
       pip install flask anthropic openai google-generativeai pycryptodome
       export NOVA_API_PROVIDER="anthropic"  # ou "openai", "google", "custom"
       export NOVA_API_KEY="ta_clé_api"
       # Optionnel: export NOVA_API_BASE="https://api.custom.com/v1" (pour endpoints compatibles OpenAI)
       # Optionnel: export NOVA_MODEL="claude-3-5-sonnet-20241022" (modèle spécifique)
       python agent_nova.py
       -> ouvre http://127.0.0.1:5000 dans ton navigateur

  B) Ligne de commande (comme l'outil Nova Safe Core d'origine) :
       python agent_nova.py analyze mon_script.py
       python agent_nova.py scan mon_dossier/
       python agent_nova.py new-project mon_projet
       python agent_nova.py encrypt fichier.txt fichier.txt.nsafe
       python agent_nova.py decrypt fichier.txt.nsafe fichier.txt

Garanties de conception (conservées) :
- Aucune porte dérobée, aucune injection de processus, aucune persistance
  cachée, aucune auto-réplication.
- `analyze` et `scan` ne font JAMAIS exec()/eval() du code inspecté : tout
  passe par `ast.parse`.
- Le chiffrement n'altère jamais le fichier source (AES-256-GCM, intégrité
  vérifiée).
- L'agent IA ne peut PAS chiffrer/déchiffrer pour toi : ces opérations
  restent manuelles, pour qu'aucun mot de passe ne transite par l'IA.
"""

from __future__ import annotations

import argparse
import ast
import getpass
import hashlib
import json
import os
import secrets
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# =============================================================================
# CONFIGURATION MULTI-FOURNISSEURS
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration pour un fournisseur d'API LLM."""
    name: str
    env_key: str
    env_base: str | None = None
    env_model: str | None = None
    default_model: str = ""
    default_base: str | None = None
    requires_base: bool = False


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="Anthropic",
        env_key="ANTHROPIC_API_KEY",
        env_base="ANTHROPIC_API_BASE",
        env_model="ANTHROPIC_MODEL",
        default_model="claude-3-5-sonnet-20241022",
    ),
    "openai": ProviderConfig(
        name="OpenAI",
        env_key="OPENAI_API_KEY",
        env_base="OPENAI_API_BASE",
        env_model="OPENAI_MODEL",
        default_model="gpt-4o",
        default_base="https://api.openai.com/v1",
    ),
    "google": ProviderConfig(
        name="Google (Gemini)",
        env_key="GOOGLE_API_KEY",
        env_base="GOOGLE_API_BASE",
        env_model="GOOGLE_MODEL",
        default_model="gemini-1.5-pro",
    ),
    "mistral": ProviderConfig(
        name="Mistral AI",
        env_key="MISTRAL_API_KEY",
        env_base="MISTRAL_API_BASE",
        env_model="MISTRAL_MODEL",
        default_model="mistral-large-latest",
        default_base="https://api.mistral.ai/v1",
    ),
    "groq": ProviderConfig(
        name="Groq",
        env_key="GROQ_API_KEY",
        env_base="GROQ_API_BASE",
        env_model="GROQ_MODEL",
        default_model="llama-3.3-70b-versatile",
        default_base="https://api.groq.com/openai/v1",
    ),
    "ollama": ProviderConfig(
        name="Ollama (local)",
        env_key="OLLAMA_API_KEY",  # usually not required
        env_base="OLLAMA_API_BASE",
        env_model="OLLAMA_MODEL",
        default_model="llama3.1",
        default_base="http://localhost:11434/v1",
        requires_base=False,
    ),
    "custom": ProviderConfig(
        name="Custom (OpenAI-compatible)",
        env_key="NOVA_API_KEY",
        env_base="NOVA_API_BASE",
        env_model="NOVA_MODEL",
        default_model="",
        default_base="",
        requires_base=True,
    ),
}

# Configuration unifiée via variables d'environnement NOVA_*
NOVA_PROVIDER = os.environ.get("NOVA_API_PROVIDER", "anthropic").lower()
NOVA_API_KEY = os.environ.get("NOVA_API_KEY")
NOVA_API_BASE = os.environ.get("NOVA_API_BASE")
NOVA_MODEL = os.environ.get("NOVA_MODEL")


def get_active_provider() -> ProviderConfig:
    """Retourne la configuration du fournisseur actif."""
    if NOVA_PROVIDER in PROVIDERS:
        return PROVIDERS[NOVA_PROVIDER]
    return PROVIDERS["custom"]


def resolve_config() -> tuple[str, str | None, str]:
    """
    Résout la configuration finale (clé API, base URL, modèle) en fusionnant :
    1. Variables NOVA_* (priorité haute, universelles)
    2. Variables spécifiques au fournisseur (ex: ANTHROPIC_API_KEY)
    3. Valeurs par défaut du fournisseur
    """
    provider = get_active_provider()

    # Clé API
    api_key = NOVA_API_KEY or os.environ.get(provider.env_key)
    if not api_key and provider.name != "Ollama (local)":
        raise RuntimeError(
            f"Clé API manquante. Définis {provider.env_key} ou NOVA_API_KEY.\n"
            f"Fournisseur actuel : {provider.name} ({NOVA_PROVIDER})"
        )

    # Base URL
    base_url = NOVA_API_BASE or os.environ.get(provider.env_base) or provider.default_base

    # Modèle
    model = NOVA_MODEL or os.environ.get(provider.env_model) or provider.default_model
    if not model:
        raise RuntimeError(
            f"Modèle non défini. Définis {provider.env_model} ou NOVA_MODEL, "
            f"ou choisis un fournisseur avec un modèle par défaut."
        )

    return api_key, base_url, model


# =============================================================================
# PARTIE 1 : NOVA SAFE CORE (chiffrement, analyse statique, scaffolding)
# =============================================================================

MAGIC = b"NOVASAFE2"
SALT_SIZE = 16
NONCE_SIZE = 12
KEY_SIZE = 32
KDF_ITERATIONS = 600_000

SUSPICIOUS_CALLS = {
    "eval": "exécution dynamique de code",
    "exec": "exécution dynamique de code",
    "os.system": "exécution de commande shell",
    "subprocess.Popen": "lancement de processus",
    "subprocess.run": "lancement de processus",
    "subprocess.call": "lancement de processus",
    "pickle.loads": "désérialisation non sûre",
    "pickle.load": "désérialisation non sûre",
    "marshal.loads": "désérialisation non sûre",
    "ctypes.windll": "accès bas niveau Windows (API native)",
    "winreg.SetValueEx": "modification du registre Windows",
    "requests.get": "requête réseau sortante",
    "urllib.request.urlopen": "requête réseau sortante",
    "socket.socket": "ouverture de socket réseau",
}


def derive_key(password: str, salt: bytes) -> bytes:
    """Dérive une clé AES-256 depuis un mot de passe avec PBKDF2-HMAC-SHA256."""
    if not password:
        raise ValueError("Le mot de passe ne peut pas être vide.")
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, KDF_ITERATIONS, dklen=KEY_SIZE
    )


def _require_aes():
    try:
        from Crypto.Cipher import AES
        return AES
    except ImportError as exc:
        raise RuntimeError(
            "Dépendance manquante. Installe-la avec : python -m pip install pycryptodome"
        ) from exc


def encrypt_file(source: Path, destination: Path, password: str) -> Path:
    """Chiffre un fichier avec AES-256-GCM et vérification d'intégrité."""
    AES = _require_aes()
    if not source.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Le fichier de sortie doit être différent du fichier source.")

    data = source.read_bytes()
    salt = secrets.token_bytes(SALT_SIZE)
    nonce = secrets.token_bytes(NONCE_SIZE)
    key = derive_key(password, salt)

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(MAGIC + salt + nonce + tag + ciphertext)
    return destination


def decrypt_file(source: Path, destination: Path, password: str) -> Path:
    """Déchiffre un fichier NOVASAFE2 et refuse toute donnée altérée."""
    AES = _require_aes()
    if not source.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Le fichier de sortie doit être différent du fichier source.")

    blob = source.read_bytes()
    minimum = len(MAGIC) + SALT_SIZE + NONCE_SIZE + 16
    if len(blob) < minimum or not blob.startswith(MAGIC):
        raise ValueError("Format invalide : ce fichier n'est pas un fichier NOVASAFE2.")

    offset = len(MAGIC)
    salt = blob[offset:offset + SALT_SIZE]; offset += SALT_SIZE
    nonce = blob[offset:offset + NONCE_SIZE]; offset += NONCE_SIZE
    tag = blob[offset:offset + 16]; offset += 16
    ciphertext = blob[offset:]

    key = derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(plaintext)
    return destination


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def encrypt_tree(source: Path, destination: Path, password: str, suffix: str = ".nsafe") -> list[Path]:
    """Chiffre récursivement un fichier ou un dossier entier."""
    outputs: list[Path] = []
    if source.is_file():
        out = destination if destination.suffix else destination / (source.name + suffix)
        outputs.append(encrypt_file(source, out, password))
        return outputs
    for f in _iter_files(source):
        rel = f.relative_to(source)
        out = destination / rel.parent / (rel.name + suffix)
        outputs.append(encrypt_file(f, out, password))
    return outputs


def decrypt_tree(source: Path, destination: Path, password: str, suffix: str = ".nsafe") -> list[Path]:
    """Déchiffre récursivement un fichier ou un dossier entier."""
    outputs: list[Path] = []
    if source.is_file():
        name = source.name[:-len(suffix)] if source.name.endswith(suffix) else source.name + ".dec"
        out = destination if destination.suffix else destination / name
        outputs.append(decrypt_file(source, out, password))
        return outputs
    for f in _iter_files(source):
        if f.suffix != suffix:
            continue
        rel = f.relative_to(source)
        out_name = rel.name[: -len(suffix)]
        out = destination / rel.parent / out_name
        outputs.append(decrypt_file(f, out, password))
    return outputs


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruit 'a.b.c' pour un Attribute/Name chaîné, sinon None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def analyze_python_file(path: Path, max_line_length: int = 100) -> dict[str, Any]:
    """Analyse la syntaxe, la qualité et les signaux de sécurité, sans exécuter le fichier."""
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    result: dict[str, Any] = {
        "fichier": str(path),
        "taille_octets": path.stat().st_size,
        "lignes": len(lines),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        result.update({
            "syntaxe_valide": False,
            "erreur": {"message": exc.msg, "ligne": exc.lineno, "colonne": exc.offset},
        })
        return result

    functions, classes, imports = [], [], []
    suspects: list[dict[str, Any]] = []
    docstring_manquante = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            if ast.get_docstring(node) is None and not node.name.startswith("_"):
                docstring_manquante.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = _dotted_name(node.func)
            if name in SUSPICIOUS_CALLS:
                suspects.append({
                    "appel": name,
                    "ligne": node.lineno,
                    "raison": SUSPICIOUS_CALLS[name],
                })

    lignes_trop_longues = [i + 1 for i, l in enumerate(lines) if len(l) > max_line_length]

    result.update({
        "syntaxe_valide": True,
        "fonctions": functions,
        "classes": classes,
        "imports": sorted(set(imports)),
        "nombre_fonctions": len(functions),
        "nombre_classes": len(classes),
        "appels_suspects": suspects,
        "fonctions_sans_docstring": docstring_manquante,
        "lignes_trop_longues": lignes_trop_longues,
        "niveau_risque": "élevé" if len(suspects) >= 3 else ("moyen" if suspects else "faible"),
    })
    return result


def scan_directory(root: Path, max_line_length: int = 100) -> dict[str, Any]:
    """Audite tous les fichiers .py d'un dossier et agrège les résultats."""
    if not root.is_dir():
        raise NotADirectoryError(f"Dossier introuvable : {root}")

    reports = []
    total_suspects = 0
    for f in sorted(root.rglob("*.py")):
        try:
            report = analyze_python_file(f, max_line_length=max_line_length)
        except (SyntaxError, UnicodeDecodeError) as exc:
            report = {"fichier": str(f), "syntaxe_valide": False, "erreur": str(exc)}
        reports.append(report)
        total_suspects += len(report.get("appels_suspects", []))

    return {
        "dossier": str(root),
        "fichiers_analyses": len(reports),
        "appels_suspects_total": total_suspects,
        "rapports": reports,
    }


def create_project(destination: Path, name: str | None = None) -> Path:
    """Crée un squelette de projet Python propre, sans lancement automatique."""
    destination.mkdir(parents=True, exist_ok=True)
    project_name = name or destination.name

    files = {
        "main.py": (
            '"""Point d\'entrée du projet."""\n\n'
            'def main() -> None:\n'
            f'    print("Bienvenue dans {project_name}.")\n\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        ),
        "README.md": (
            f"# {project_name}\n\n"
            "Projet local. Examinez toujours le code avant de l'exécuter.\n\n"
            "## Lancement\n\n```\npython main.py\n```\n\n"
            "## Tests\n\n```\npython -m pytest tests/\n```\n"
        ),
        ".gitignore": "__pycache__/\n*.pyc\n.venv/\n*.nsafe\n.env\n",
        "requirements.txt": "# Ajoute ici tes dépendances, une par ligne\n",
        "tests/test_main.py": (
            "def test_placeholder():\n"
            "    assert True  # Remplace par de vrais tests\n"
        ),
    }

    existing = [n for n in files if (destination / n).exists()]
    if existing:
        raise FileExistsError(
            f"Ces fichiers existent déjà, rien n'a été écrasé : {', '.join(existing)}"
        )

    for rel_path, content in files.items():
        full = destination / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    return destination


# =============================================================================
# PARTIE 2 : AGENT IA — ABSTRACTION MULTI-FOURNISSEURS
# =============================================================================

SYSTEM_PROMPT = (
    "Tu es l'Agent IA NOVA, l'assistant personnel d'Henok, entrepreneur digital "
    "basé au Togo (marque NOVA DIGITAL). Tu l'aides à gérer ses projets de code "
    "en appelant les outils à ta disposition : analyser un fichier Python, "
    "auditer un dossier de code, ou créer un squelette de nouveau projet. "
    "Réponds toujours en français, de façon claire et concise, comme si tu "
    "t'adressais à quelqu'un qui débute en développement. Quand un outil "
    "renvoie un résultat technique, explique-le en langage simple."
)

TOOLS = [
    {
        "name": "analyze_file",
        "description": (
            "Analyse un fichier Python SANS l'exécuter : liste ses fonctions, "
            "classes, imports, et signale les appels potentiellement dangereux."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chemin": {"type": "string", "description": "Chemin du fichier .py à analyser"}
            },
            "required": ["chemin"],
        },
    },
    {
        "name": "scan_folder",
        "description": (
            "Audite tous les fichiers .py d'un dossier (récursivement) et "
            "renvoie un résumé des appels suspects trouvés dans chaque fichier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chemin": {"type": "string", "description": "Chemin du dossier à auditer"}
            },
            "required": ["chemin"],
        },
    },
    {
        "name": "new_project",
        "description": (
            "Crée un squelette de nouveau projet Python (main.py, README.md, "
            ".gitignore, requirements.txt, tests/) dans le dossier indiqué."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chemin": {"type": "string", "description": "Dossier de destination du projet"},
                "nom": {"type": "string", "description": "Nom du projet (optionnel)"},
            },
            "required": ["chemin"],
        },
    },
]


class LLMClient(ABC):
    """Interface commune pour tous les fournisseurs LLM."""

    @abstractmethod
    def create_message(
        self,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict],
        messages: list[dict],
    ) -> Any:
        """Envoie une requête et retourne la réponse brute du fournisseur."""
        pass

    @abstractmethod
    def extract_content(self, response: Any) -> tuple[list[str], list[dict]]:
        """
        Extrait le texte et les tool_use de la réponse.
        Retourne (text_blocks, tool_uses).
        """
        pass

    @abstractmethod
    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        """Formate les résultats d'outils pour le fournisseur."""
        pass


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def create_message(self, model: str, max_tokens: int, system: str,
                       tools: list[dict], messages: list[dict]) -> Any:
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

    def extract_content(self, response: Any) -> tuple[list[str], list[dict]]:
        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_uses = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in response.content if b.type == "tool_use"
        ]
        return text_blocks, tool_uses

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        return tool_results  # Anthropic format natively matches


class OpenAICompatibleClient(LLMClient):
    """Client pour OpenAI, Mistral, Groq, Ollama, et endpoints compatibles."""

    def __init__(self, api_key: str, base_url: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def create_message(self, model: str, max_tokens: int, system: str,
                       tools: list[dict], messages: list[dict]) -> Any:
        # Convertir le format messages pour OpenAI
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        # Convertir tools au format OpenAI
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
            })

        return self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
        )

    def extract_content(self, response: Any) -> tuple[list[str], list[dict]]:
        choice = response.choices[0]
        msg = choice.message
        text_blocks = [msg.content] if msg.content else []
        tool_uses = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_uses.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
        return text_blocks, tool_uses

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        # Format OpenAI pour tool results
        formatted = []
        for tr in tool_results:
            formatted.append({
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": tr["content"],
            })
        return formatted


class GoogleClient(LLMClient):
    """Client pour Google Gemini (avec function calling)."""

    def __init__(self, api_key: str, base_url: str | None = None):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_name = None  # sera défini à l'appel

    def create_message(self, model: str, max_tokens: int, system: str,
                       tools: list[dict], messages: list[dict]) -> Any:
        self.model_name = model
        # Convertir tools au format Gemini
        gemini_tools = []
        for t in tools:
            gemini_tools.append({
                "function_declarations": [{
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }]
            })

        model = self.genai.GenerativeModel(
            model_name=model,
            tools=gemini_tools if gemini_tools else None,
            system_instruction=system if system else None,
        )

        # Convertir messages
        gemini_messages = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            content = m["content"]
            if isinstance(content, list):
                # Gérer les tool_results
                parts = []
                for item in content:
                    if item.get("type") == "tool_result":
                        parts.append({
                            "function_response": {
                                "name": item.get("name", "unknown"),
                                "response": {"result": item["content"]},
                            }
                        })
                    elif item.get("type") == "text":
                        parts.append({"text": item["text"]})
                gemini_messages.append({"role": role, "parts": parts})
            else:
                gemini_messages.append({"role": role, "parts": [{"text": str(content)}]})

        chat = model.start_chat(history=gemini_messages[:-1])
        return chat.send_message(gemini_messages[-1]["parts"])

    def extract_content(self, response: Any) -> tuple[list[str], list[dict]]:
        text_blocks = []
        tool_uses = []
        for part in response.parts:
            if part.text:
                text_blocks.append(part.text)
            if part.function_call:
                fc = part.function_call
                tool_uses.append({
                    "id": fc.id or "call_" + secrets.token_hex(8),
                    "name": fc.name,
                    "input": dict(fc.args),
                })
        return text_blocks, tool_uses

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        # Pour Gemini, on renvoie dans le prochain message utilisateur
        parts = []
        for tr in tool_results:
            parts.append({
                "function_response": {
                    "name": tr.get("name", "unknown"),
                    "response": {"result": tr["content"]},
                }
            })
        return [{"role": "user", "parts": parts}]


def create_llm_client(provider: str, api_key: str, base_url: str | None) -> LLMClient:
    """Factory pour créer le client approprié selon le fournisseur."""
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(api_key, base_url)
    elif provider in ("openai", "mistral", "groq", "ollama", "custom"):
        return OpenAICompatibleClient(api_key, base_url)
    elif provider == "google":
        return GoogleClient(api_key, base_url)
    else:
        # Par défaut, essayer OpenAI-compatible
        return OpenAICompatibleClient(api_key, base_url)


_lock = threading.Lock()
_state: dict = {"messages": []}


def run_tool(name: str, tool_input: dict) -> str:
    try:
        if name == "analyze_file":
            report = analyze_python_file(Path(tool_input["chemin"]))
            return json.dumps(report, ensure_ascii=False)

        if name == "scan_folder":
            report = scan_directory(Path(tool_input["chemin"]))
            resume = {
                "dossier": report["dossier"],
                "fichiers_analyses": report["fichiers_analyses"],
                "appels_suspects_total": report["appels_suspects_total"],
                "fichiers_avec_suspects": [
                    r["fichier"] for r in report["rapports"] if r.get("appels_suspects")
                ],
            }
            return json.dumps(resume, ensure_ascii=False)

        if name == "new_project":
            path = create_project(Path(tool_input["chemin"]), name=tool_input.get("nom"))
            return json.dumps({"statut": "créé", "chemin": str(path)}, ensure_ascii=False)

        return json.dumps({"erreur": f"Outil inconnu : {name}"})

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"erreur": str(exc)}, ensure_ascii=False)


def process_message(user_text: str) -> tuple[str, list[dict]]:
    """Envoie le message au LLM configuré, exécute les outils demandés, et renvoie la réponse finale."""
    api_key, base_url, model = resolve_config()
    provider = get_active_provider()
    client = create_llm_client(NOVA_PROVIDER, api_key, base_url)

    messages = _state["messages"]
    messages.append({"role": "user", "content": user_text})

    tool_log: list[dict] = []
    final_text_parts: list[str] = []

    while True:
        response = client.create_message(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Normaliser la réponse pour l'historique
        text_blocks, tool_uses = client.extract_content(response)

        # Stocker la réponse de l'assistant
        if NOVA_PROVIDER == "google":
            # Format Gemini
            assistant_content = []
            for t in text_blocks:
                assistant_content.append({"type": "text", "text": t})
            for tu in tool_uses:
                assistant_content.append({
                    "type": "function_call",
                    "id": tu["id"],
                    "name": tu["name"],
                    "args": tu["input"],
                })
            messages.append({"role": "model", "content": assistant_content})
        else:
            # Format Anthropic / OpenAI
            assistant_content = []
            for t in text_blocks:
                assistant_content.append({"type": "text", "text": t})
            for tu in tool_uses:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tu["id"],
                    "name": tu["name"],
                    "input": tu["input"],
                })
            messages.append({"role": "assistant", "content": assistant_content})

        final_text_parts.extend(text_blocks)

        if not tool_uses:
            break

        # Exécuter les outils
        tool_results = []
        for call in tool_uses:
            result = run_tool(call["name"], call["input"])
            tool_log.append({"name": call["name"], "input": call["input"], "result": result})

            if NOVA_PROVIDER == "anthropic":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result,
                })
            elif NOVA_PROVIDER == "google":
                tool_results.append({
                    "type": "tool_result",
                    "name": call["name"],
                    "tool_use_id": call["id"],
                    "content": result,
                })
            else:
                # OpenAI compatible
                tool_results.append({
                    "tool_use_id": call["id"],
                    "name": call["name"],
                    "content": result,
                })

        # Ajouter les résultats d'outils
        formatted_results = client.format_tool_results(tool_results)
        if NOVA_PROVIDER == "google":
            # Gemini attend un message utilisateur avec function_response
            messages.extend(formatted_results)
        else:
            messages.append({"role": "user", "content": formatted_results})

    return "\n\n".join(final_text_parts).strip(), tool_log


# =============================================================================
# PARTIE 3 : INTERFACE WEB (page de chat embarquée + serveur Flask)
# =============================================================================


HTML_PAGE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGENT IA NOVA · Assistant de code</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f7fb;--p:#fff;--ink:#0b1a2b;--dim:#6b7c93;--ln:#d5deea;--a:#0a84ff;--b:#ff5a1f;--ok:#12b76a;--er:#e5383b;--m:'Share Tech Mono',ui-monospace,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{font:16px/1.5 Rajdhani,system-ui,sans-serif;color:var(--ink);background:var(--bg);min-height:100vh;display:flex;flex-direction:column}
body::before{content:"";position:fixed;inset:0;z-index:-1;background:linear-gradient(var(--ln) 1px,transparent 1px) 0 0/40px 40px,linear-gradient(90deg,var(--ln) 1px,transparent 1px) 0 0/40px 40px;opacity:.5}
body::after{content:"";position:fixed;top:0;left:0;right:0;height:6px;z-index:25;background:repeating-linear-gradient(90deg,var(--ink) 0 1px,transparent 1px 10px);opacity:.25;pointer-events:none}

header{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--ink);padding:10px 18px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;justify-content:space-between}
.logo{font:800 20px Orbitron;letter-spacing:.1em}.logo b{color:var(--a)}.logo i{display:block;font:10px var(--m);letter-spacing:.3em;color:var(--dim);font-style:normal}
.hb{display:flex;gap:6px;align-items:center}
.chip{font:11px var(--m);padding:3px 9px;border:1px solid var(--ok);color:var(--ok);text-transform:uppercase;letter-spacing:.05em}
.chip.off{border-color:var(--er);color:var(--er)}
button,.btn{font:600 12px var(--m);letter-spacing:.1em;text-transform:uppercase;padding:8px 13px;border:1px solid var(--ink);background:#fff;color:var(--ink);cursor:pointer}
button:hover:not(:disabled),.btn:hover{background:var(--ink);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
.sm{padding:5px 9px;font-size:10px}

main{flex:1;max-width:900px;width:100%;margin:0 auto;padding:22px 16px 160px;display:flex;flex-direction:column}
.hero{background:var(--p);border:1px solid var(--ln);padding:18px;margin-bottom:16px;position:relative}
.hero::before,.hero::after{content:"";position:absolute;width:14px;height:14px;border:2px solid var(--a)}
.hero::before{top:-1px;left:-1px;border-right:0;border-bottom:0}.hero::after{bottom:-1px;right:-1px;border-left:0;border-top:0}
.lab{font:11px var(--m);letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
h2{font:600 22px Orbitron;letter-spacing:.04em;margin:4px 0 6px}h2 b{color:var(--a)}
p.sub{color:var(--dim);max-width:620px}

#thread{flex:1;display:flex;flex-direction:column;gap:12px;margin-bottom:14px}
.msg{max-width:78%;padding:11px 14px;border:1px solid var(--ln);background:#fff;position:relative}
.msg.user{align-self:flex-end;border-color:var(--a);background:#eef6ff}
.msg.agent{align-self:flex-start;border-color:var(--ink)}
.msg .who{font:10px var(--m);letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin-bottom:4px;display:block}
.msg.user .who{color:var(--a)}
.msg p{white-space:pre-wrap}
.msg.pending{color:var(--dim);font-style:italic}
.toolline{font:11px var(--m);color:var(--b);border-left:2px solid var(--b);padding:2px 8px;margin-top:6px}

#inputBar{position:fixed;left:0;right:0;bottom:0;background:rgba(244,247,251,.95);backdrop-filter:blur(8px);border-top:1px solid var(--ink);padding:12px 16px;z-index:30}
#inputRow{max-width:900px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;font:14px/1.4 Rajdhani,sans-serif;padding:10px;border:1px solid var(--ln);background:#f9fbfd;color:var(--ink);outline:0;resize:none;min-height:46px;max-height:140px}
textarea:focus{border-color:var(--a);background:#fff}
.big{background:var(--a);border-color:var(--a);color:#fff;padding:0 22px;font:600 12px var(--m);letter-spacing:.14em;text-transform:uppercase;min-height:46px}
.big:hover:not(:disabled){background:#0066cc;border-color:#0066cc}

#con{position:fixed;right:16px;bottom:78px;width:320px;max-height:40vh;z-index:40;background:#0b1a2b;color:#bfe3ff;transform:translateX(360px);transition:transform .25s;display:flex;flex-direction:column;border:1px solid #26405c}
#con.on{transform:none}
#con .hd{display:flex;justify-content:space-between;align-items:center;padding:6px 10px;border-bottom:1px solid #26405c;font:11px var(--m);letter-spacing:.08em}
#con .hd button{background:transparent;border:1px solid #4a6a8a;color:#bfe3ff;padding:2px 8px;font-size:9px}
#lb{flex:1;overflow:auto;padding:8px 10px;font:11px/1.5 var(--m)}
.lg{border-bottom:1px dashed #26405c;padding:4px 0}
.lg b{color:#7dffb5}

.provider-badge{font:10px var(--m);padding:2px 8px;border:1px solid var(--dim);color:var(--dim);border-radius:3px;margin-left:8px}

@media(max-width:700px){
  .msg{max-width:92%}
  #con{right:8px;left:8px;width:auto}
}
</style></head><body>

<header>
  <div class="logo">AGENT <b>IA</b><i>NOVA DIGITAL · ASSISTANT DE CODE</i></div>
  <div class="hb">
    <span class="chip" id="statusChip">Vérification…</span>
    <span class="provider-badge" id="providerBadge">Provider: —</span>
    <button class="sm" id="toggleConsole">Journal</button>
    <button class="sm" id="resetBtn">Nouvelle session</button>
  </div>
</header>

<main>
  <div class="hero">
    <div class="lab">Assistant conversationnel</div>
    <h2>Parle à ton <b>Agent IA</b></h2>
    <p class="sub">Demande-lui d'analyser un fichier Python, d'auditer un dossier de code, ou de créer un squelette de nouveau projet. Il s'exécute entièrement en local.</p>
  </div>

  <div id="thread"></div>
</main>

<div id="inputBar">
  <div id="inputRow">
    <textarea id="input" placeholder="Écris ton message… (Entrée pour envoyer, Maj+Entrée pour une nouvelle ligne)"></textarea>
    <button class="big" id="sendBtn">Envoyer</button>
  </div>
</div>

<div id="con">
  <div class="hd"><span>JOURNAL D'EXÉCUTION</span><button id="closeConsole">✕</button></div>
  <div id="lb"></div>
</div>

<script>
const thread = document.getElementById('thread');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const statusChip = document.getElementById('statusChip');
const providerBadge = document.getElementById('providerBadge');
const con = document.getElementById('con');
const lb = document.getElementById('lb');

function addMsg(who, text, pending=false){
  const d = document.createElement('div');
  d.className = 'msg ' + (who==='user'?'user':'agent') + (pending?' pending':'');
  d.innerHTML = '<span class="who">'+(who==='user'?'Toi':'Agent NOVA')+'</span><p></p>';
  d.querySelector('p').textContent = text;
  thread.appendChild(d);
  window.scrollTo(0, document.body.scrollHeight);
  return d;
}

function logLine(text){
  const l = document.createElement('div');
  l.className = 'lg';
  l.innerHTML = text;
  lb.appendChild(l);
  lb.scrollTop = lb.scrollHeight;
}

async function checkStatus(){
  try{
    const r = await fetch('/api/status');
    const d = await r.json();
    if(d.cle_api_presente && d.client_ok){
      statusChip.textContent = 'Agent prêt';
      statusChip.className = 'chip';
    } else {
      statusChip.textContent = d.cle_api_presente ? 'Client non configuré' : 'Clé API manquante';
      statusChip.className = 'chip off';
    }
    providerBadge.textContent = 'Provider: ' + (d.fournisseur || 'inconnu');
    if(d.modele) providerBadge.textContent += ' · ' + d.modele;
  }catch(e){
    statusChip.textContent = 'Serveur injoignable';
    statusChip.className = 'chip off';
  }
}

async function send(){
  const text = input.value.trim();
  if(!text) return;
  input.value = '';
  addMsg('user', text);
  const pending = addMsg('agent', 'Réflexion…', true);
  sendBtn.disabled = true;

  try{
    const r = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });
    const d = await r.json();
    pending.remove();
    if(d.erreur){
      addMsg('agent', 'Erreur : ' + d.erreur);
      logLine('<b>ERREUR</b> ' + d.erreur);
    } else {
      addMsg('agent', d.reponse || '(réponse vide)');
      (d.appels_outils||[]).forEach(t=>{
        logLine('<b>'+t.name+'</b>('+JSON.stringify(t.input)+')');
      });
    }
  }catch(e){
    pending.remove();
    addMsg('agent', 'Erreur réseau : ' + e.message);
  }
  sendBtn.disabled = false;
  input.focus();
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
});
document.getElementById('resetBtn').addEventListener('click', async ()=>{
  await fetch('/api/reset', {method:'POST'});
  thread.innerHTML = '';
  lb.innerHTML = '';
  addMsg('agent', 'Nouvelle session démarrée. Comment puis-je t\'aider ?');
});
document.getElementById('toggleConsole').addEventListener('click', ()=>con.classList.add('on'));
document.getElementById('closeConsole').addEventListener('click', ()=>con.classList.remove('on'));

checkStatus();
addMsg('agent', 'Bonjour ! Je suis ton Agent IA NOVA. Demande-moi d\'analyser un fichier, d\'auditer un dossier, ou de créer un nouveau projet.');
</script>
</body></html>
"""


def _build_flask_app():
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)

    @app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    @app.route("/api/status")
    def api_status():
        provider = get_active_provider()
        api_key_present = False
        client_ok = False
        model = ""
        try:
            api_key, base_url, model = resolve_config()
            api_key_present = bool(api_key)
            client_ok = True
        except RuntimeError:
            pass

        return jsonify({
            "fournisseur": provider.name,
            "provider_key": NOVA_PROVIDER,
            "modele": model,
            "base_url": base_url,
            "cle_api_presente": api_key_present,
            "client_ok": client_ok,
            "messages_en_memoire": len(_state["messages"]),
            "fournisseurs_disponibles": list(PROVIDERS.keys()),
        })

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("message") or "").strip()
        if not text:
            return jsonify({"erreur": "Message vide."}), 400
        with _lock:
            try:
                reply, tool_log = process_message(text)
            except Exception as exc:  # noqa: BLE001
                return jsonify({"erreur": str(exc)}), 500
        return jsonify({"reponse": reply, "appels_outils": tool_log})

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        with _lock:
            _state["messages"] = []
        return jsonify({"statut": "ok"})

    return app


def run_web(host: str = "127.0.0.1", port: int = 5000) -> None:
    try:
        app = _build_flask_app()
    except ImportError:
        print("Dépendance manquante. Installe-la avec : pip install flask")
        sys.exit(1)
    provider = get_active_provider()
    print(f"Agent IA NOVA — interface web sur http://{host}:{port}")
    print(f"Fournisseur: {provider.name} ({NOVA_PROVIDER})")
    api_key, base_url, model = resolve_config()
    print(f"Modèle: {model}")
    if base_url:
        print(f"Base URL: {base_url}")
    app.run(host=host, port=port, debug=False)


# =============================================================================
# CLI (mode ligne de commande, hérité de Nova Safe Core)
# =============================================================================

def ask_password(confirm: bool = False) -> str:
    password = getpass.getpass("Mot de passe : ")
    if confirm:
        repeated = getpass.getpass("Confirme le mot de passe : ")
        if password != repeated:
            raise ValueError("Les mots de passe ne correspondent pas.")
    return password


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent IA NOVA : interface web par défaut, ou outils en ligne de commande."
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default=None,
        help="Fournisseur LLM à utiliser (défaut: NOVA_API_PROVIDER ou anthropic)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modèle spécifique à utiliser (écrase NOVA_MODEL et la valeur par défaut du fournisseur)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Clé API (écrase les variables d'environnement)"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="URL de base de l'API (pour endpoints compatibles OpenAI)"
    )

    sub = parser.add_subparsers(dest="command")

    web = sub.add_parser("web", help="Lancer l'interface web (défaut si aucune commande).")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=5000)

    enc = sub.add_parser("encrypt", help="Chiffrer un fichier ou un dossier entier.")
    enc.add_argument("source", type=Path)
    enc.add_argument("destination", type=Path)

    dec = sub.add_parser("decrypt", help="Déchiffrer un fichier ou un dossier entier.")
    dec.add_argument("source", type=Path)
    dec.add_argument("destination", type=Path)

    analyze = sub.add_parser("analyze", help="Analyser un fichier Python sans l'exécuter.")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--max-line-length", type=int, default=100)

    scan = sub.add_parser("scan", help="Auditer tous les fichiers .py d'un dossier.")
    scan.add_argument("source", type=Path)
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--max-line-length", type=int, default=100)

    project = sub.add_parser("new-project", help="Créer un squelette de projet complet.")
    project.add_argument("destination", type=Path)
    project.add_argument("--name", type=str, default=None)

    return parser


def _print_analysis(report: dict[str, Any]) -> None:
    print(f"Fichier : {report['fichier']}")
    print(f"Taille : {report['taille_octets']} octets | Lignes : {report['lignes']}")
    print(f"SHA-256 : {report['sha256']}")
    print(f"Syntaxe valide : {'oui' if report['syntaxe_valide'] else 'non'}")
    if not report["syntaxe_valide"]:
        print(f"Erreur : {report['erreur']}")
        return
    print(f"Classes : {', '.join(report['classes']) or 'aucune'}")
    print(f"Fonctions : {', '.join(report['fonctions']) or 'aucune'}")
    print(f"Imports : {', '.join(report['imports']) or 'aucun'}")
    print(f"Niveau de risque : {report['niveau_risque']}")
    if report["appels_suspects"]:
        print("Appels à vérifier :")
        for s in report["appels_suspects"]:
            print(f"  - ligne {s['ligne']} : {s['appel']} ({s['raison']})")
    if report["fonctions_sans_docstring"]:
        print(f"Fonctions sans docstring : {', '.join(report['fonctions_sans_docstring'])}")
    if report["lignes_trop_longues"]:
        print(f"Lignes trop longues : {len(report['lignes_trop_longues'])}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Appliquer les arguments CLI pour surcharger la config
    if args.provider:
        os.environ["NOVA_API_PROVIDER"] = args.provider
    if args.model:
        os.environ["NOVA_MODEL"] = args.model
    if args.api_key:
        os.environ["NOVA_API_KEY"] = args.api_key
    if args.api_base:
        os.environ["NOVA_API_BASE"] = args.api_base

    command = args.command or "web"

    try:
        if command == "web":
            run_web(host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 5000))
            return 0

        elif command == "encrypt":
            password = ask_password(confirm=True)
            outputs = encrypt_tree(args.source, args.destination, password)
            for o in outputs:
                print(f"Chiffré : {o}")
            print(f"{len(outputs)} fichier(s) chiffré(s).")

        elif command == "decrypt":
            password = ask_password()
            outputs = decrypt_tree(args.source, args.destination, password)
            for o in outputs:
                print(f"Déchiffré : {o}")
            print(f"{len(outputs)} fichier(s) déchiffré(s).")

        elif command == "analyze":
            report = analyze_python_file(args.source, max_line_length=args.max_line_length)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                _print_analysis(report)
            return 0 if report["syntaxe_valide"] else 2

        elif command == "scan":
            result = scan_directory(args.source, max_line_length=args.max_line_length)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"Dossier : {result['dossier']}")
                print(f"Fichiers analysés : {result['fichiers_analyses']}")
                print(f"Appels suspects au total : {result['appels_suspects_total']}")
                for report in result["rapports"]:
                    if report.get("appels_suspects") or not report.get("syntaxe_valide", True):
                        print("-" * 40)
                        _print_analysis(report)

        elif command == "new-project":
            path = create_project(args.destination, name=args.name)
            print(f"Projet créé dans : {path}")

        return 0
    except (OSError, ValueError, RuntimeError, NotADirectoryError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())