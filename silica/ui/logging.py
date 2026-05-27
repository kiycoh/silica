import logging
import sys
from typing import Any

# ANSI escape codes for formatting
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

# Italian translations/explanations mapping for standard log messages
FRIENDLY_TEMPLATES = {
    # Debug messages
    "Registered tool: %s (class=%s)": 
        "Tool registrato nel sistema: {0} (implementato dalla classe {1})",
    "Redaction failed (omitting detail): %s": 
        "Censura dati sensibili fallita (dettagli omessi per sicurezza): {0}",
    "tool_progress_callback error (swallowed): %s": 
        "Errore nel callback di avanzamento dello strumento (ignorato): {0}",
    "Agent loop iteration %d": 
        "Inizio iterazione ciclo dell'agente: {0}",
    "Creato file temporaneo di stage in: %s": 
        "Creato un file temporaneo per lo staging delle modifiche in: {0}",
    "FSM Transizione: %s -> eseguendo handler": 
        "Transizione macchina a stati: {0} -> Esecuzione del gestore associato",
    "Rebuilding FS graph index...": 
        "Aggiornamento dell'indice del File System in corso...",
    "Skipping indexing for inbox directory: %s": 
        "Salto l'indicizzazione della cartella Inbox: {0}",
    "Indexed %d notes": 
        "Indicizzazione completata: {0} note trovate e caricate",
    "CLI exec: %s": 
        "Esecuzione comando CLI: {0}",
    "CLI stderr: %s": 
        "Errore standard CLI: {0}",

    # Info messages
    "Refiner phase: %s": 
        "Fase di raffinamento: {0}",
    "Skipping already processed note: %s": 
        "Nota già elaborata precedentemente, salto: {0}",
    "Injector phase: %s": 
        "Fase di iniezione: {0}",
    "VALIDATE: no actionable ops (all skip) — short-circuit to CLEANUP": 
        "Validazione completata: nessuna modifica richiesta, passaggio diretto alla pulizia",
    "Calling Distiller LLM (payload checksum %s)": 
        "Chiamata al modello Distiller per estrarre le modifiche (checksum: {0})",
    "Distiller produced %d updates": 
        "Il Distiller ha prodotto {0} aggiornamenti",
    "Validation: l'hub '%s' non esiste. Iniettata operazione di creazione in %s": 
        "La nota hub '{0}' non esiste. Iniettata operazione di creazione in {1}",
    "Restored %s to version %d": 
        "Ripristinato file {0} alla versione {1}",
    "Rolled back created note: %s": 
        "Annullata creazione nota: {0} (rimossa durante il rollback)",
    "Rolled back created note %s (already absent)": 
        "Annullamento creazione nota {0} non necessario (già assente)",
    "Rollback complete for txn %s": 
        "Rollback completato con successo per la transazione {0}",

    # Warning messages
    "restore_version with no version number for %s — skipped": 
        "Richiesto ripristino di {0} senza un numero di versione specifico (ignorato)",
    "Failed to load recipe 'injector', using defaults: %s": 
        "Impossibile caricare la ricetta per l'injector, uso dei valori di default: {0}",
    "Failed to fetch pre-write links for %s: %s": 
        "Impossibile recuperare i link pre-scrittura per {0}: {1}",
    "Failed to write ledger: %s": 
        "Impossibile scrivere il registro delle transazioni (ledger): {0}",
    "Failed to mark rollback in ledger: %s": 
        "Impossibile registrare il rollback nel registro: {0}",
    "Failed to load recipe 'refiner', using defaults: %s": 
        "Impossibile caricare la ricetta per il refiner, uso dei valori di default: {0}",
    "Distiller provider call failed, falling back to litellm: %s": 
        "Chiamata provider Distiller fallita, ripiego su litellm standard: {0}",
    "Failed to index %s: %s": 
        "Impossibile indicizzare la nota {0}: {1}",
    "base_query not implemented in FS backend": 
        "La query di base non è implementata nel backend File System",
    "No history available for %s": 
        "Nessuna cronologia disponibile per {0}",
    "Convergence guard: tool '%s' with args %s failed consecutively. Injecting warning message.": 
        "Rilevato loop: il tool '{0}' con argomenti {1} ha fallito consecutivamente. Iniezione messaggio di avviso.",
    "Agent loop hit max iterations (%d)": 
        "Il ciclo dell'agente ha raggiunto il limite massimo di iterazioni ({0})",

    # Error messages
    "Rollback error: %s": 
        "Errore durante il rollback delle modifiche: {0}",
    "FSM Error in state %s: %s": 
        "Errore nella macchina a stati nello stato {0}: {1}",
    "Failed to take pre-write graph snapshot: %s": 
        "Impossibile salvare lo stato iniziale del grafo: {0}",
    "Failed to perform graph-diff check: %s": 
        "Impossibile confrontare le differenze del grafo: {0}",
    "Rollback partially failed: %s": 
        "Il rollback è parzialmente fallito: {0}",
    "Rollback failed: %s": 
        "Annullamento modifiche (rollback) fallito: {0}",
    "Enricher failed for task %d: %s": 
        "Fase di arricchimento fallita per l'attività {0}: {1}",
    "Distiller call hit maximum tokens limit (generation cut off)": 
        "La chiamata al Distiller ha superato il limite massimo di token consentiti",
    "Transient LLM error, retries exhausted: %s": 
        "Errore temporaneo del modello linguistico, tentativi esauriti: {0}",
    "Permanent LLM or execution error: %s": 
        "Errore permanente del modello linguistico o di esecuzione: {0}",
    "LLM call failed: %s": 
        "Chiamata al modello linguistico fallita: {0}",
    "Failed to execute or parse eval search: %s": 
        "Impossibile eseguire o analizzare la ricerca di valutazione: {0}",
    "Failed to restore %s: %s": 
        "Impossibile ripristinare la nota {0}: {1}",
    "Failed to delete created note %s during rollback: %s": 
        "Impossibile rimuovere la nota creata {0} durante il rollback: {1}",
    "Convergence guard: tool '%s' with args %s failed %d times consecutively. Aborting agent run.": 
        "Rilevato potenziale loop infinito: il tool '{0}' con argomenti {1} ha fallito {2} volte consecutive. Interruzione esecuzione.",
}

class HumanFriendlyFormatter(logging.Formatter):
    """Custom logging formatter that wraps technical log messages into a human-readable format."""

    def __init__(self) -> None:
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        # Get formatted timestamp
        time_str = self.formatTime(record, self.datefmt)

        # Style level icons
        level = record.levelno
        if level == logging.DEBUG:
            icon = f"{_CYAN}⚙{_RESET}"
        elif level == logging.INFO:
            icon = f"{_GREEN}ℹ{_RESET}"
        elif level == logging.WARNING:
            icon = f"{_YELLOW}⚠️{_RESET}"
        elif level >= logging.ERROR:
            icon = f"{_RED}❌{_RESET}"
        else:
            icon = "•"

        # Interpolate standard string representation of args
        try:
            message = record.getMessage()
        except Exception as e:
            message = f"{record.msg} (args: {record.args}) [formatting error: {e}]"

        friendly_message = None

        if record.name.startswith("silica"):
            template = record.msg
            if isinstance(template, str) and template in FRIENDLY_TEMPLATES:
                friendly_template = FRIENDLY_TEMPLATES[template]
                try:
                    # Safely inject formatted arguments into the friendly template
                    if record.args:
                        if isinstance(record.args, dict):
                            friendly_message = friendly_template.format(**record.args)
                        elif isinstance(record.args, tuple):
                            friendly_message = friendly_template.format(*record.args)
                        else:
                            friendly_message = friendly_template.format(record.args)
                    else:
                        friendly_message = friendly_template
                except Exception:
                    # Fallback to default message if formatting fails
                    pass

        # If no mapping was found or matched, use the default interpolated message
        if not friendly_message:
            friendly_message = message

        return f"  {_DIM}[{time_str}]{_RESET} {icon} {friendly_message}"
