# Silica Agent — Charter Architetturale e di Bootstrap

> Il documento genesi. L'intera codebase concreta discende da qui.
> Versione 0.1 — fondazione. Tutto ciò che segue è normativo salvo dove marcato *(aperto)*.

---

## 0. Cos'è Silica (e cosa NON è)

**Silica è un agente CLI conversazionale il cui toolset è Obsidian-nativo end-to-end.** Si apre con il comando `silica`, gira un loop agentico (leggi → ragiona → chiama tool → osserva → ripeti) esattamente come Hermes Agent, ma i tool che il modello può invocare non sono primitive generiche di filesystem — sono operazioni che parlano la lingua di un vault Obsidian, costruite sopra la CLI ufficiale di Obsidian e sopra gli script di curation già esistenti.

L'analogia corretta con Hermes va letta con precisione: Hermes **non** è "un framework specializzato per il developing". È un agente *generale* la cui competenza-dev vive nelle *skill*, non nel framework. Silica inverte deliberatamente questo: la specializzazione-Obsidian non è una skill caricata sopra un framework generale — **è cablata nel toolset stesso**. Silica non sa fare altro che curare un vault, e questo è il punto.

**Silica NON è** (non-goal espliciti, ereditati da Hermes e scartati):

- un framework agentico generale → niente toolset generico, niente `bash` libero come azione di prima classe;
- un sistema multi-piattaforma di messaggistica bidirezionale (Telegram/Discord/Slack/WhatsApp/Signal) → si tiene **solo** un sink di report monodirezionale + lo scheduler cron;
- un orchestratore di backend di esecuzione (Docker/SSH/Modal/Daytona/Vercel Sandbox) → Silica gira dove gira Obsidian;
- un protocollo di interoperabilità agente (ACP) → quella è la categoria dei *concorrenti* (vedi §1.3), non la nostra;
- un sistema di compressione di traiettorie per training, dialectic user modeling, ecc. → fuori scope.

Ciò che si tiene di un framework agentico è il **nucleo minimo**: tool-calling loop, delega a sub-agent, astrazione provider-LLM, scheduler cron, report sink. Nient'altro.

---

## 1. La filosofia — il vettore risultante

### 1.1 La frase-sorgente

> *I lavori meccanici devono essere eseguiti da script; i lavori che impiegano una CoT intensiva devono essere suddivisi in batch dal router e delegati ai sub-agent; il router orchestrerà l'intera pipeline e verificherà che i criteri di accettazione e di non-regressione siano soddisfatti.*

### 1.2 La moltiplicazione matriciale

Silica è il prodotto di tre fattori. Il vettore risultante è ciò che nessuno dei fattori, da solo, produce:

```
  [ Filosofia Hermes ]      [ Toolset Obsidian-nativo ]      [ Gate deterministici ]
  router orchestra      ×   ogni azione parla "vault"    ×   accettazione + non-regressione
  script = meccanico        (CLI ufficiale + script           misurabili sul grafo
  sub-agent = semantico      promossi a tool)
            │                          │                              │
            └──────────────────────────┴──────────────────────────────┘
                                       ▼
        SILICA = agente Obsidian-nativo con pipeline di curation a qualità garantita
```

- Solo il primo fattore → riottieni Hermes (generale, nessuna garanzia di dominio).
- Solo il secondo → riottieni i *concorrenti* (copilota reattivo, nessun gate).
- Solo il terzo → riottieni uno script CI (nessuna agenzia, nessuna comprensione semantica).

Il prodotto dei tre è la tesi del progetto.

### 1.3 Posizionamento contro i concorrenti

I due progetti di riferimento (`rait-09/obsidian-agent-client`, `m-rgba/obsidian-ai-agent`) sono **plugin Obsidian in TypeScript** che incollano una chat-shell di un agente dentro la UI via ACP o wrapping diretto del CLI dell'agente. Sono **copiloti reattivi, human-in-the-loop**: tu chiedi, l'agente edita, tu sorvegli.

Silica è un'altra specie: un **motore di curation con gate**, capace di girare *unattended*. Il moat non è la UX di chat — è la **garanzia di qualità sotto autonomia**: la prova, misurabile a livello di grafo, che un batch non ha introdotto regressioni. Non si compete sul terreno dei concorrenti.

---

## 2. Principio cardine — due consumatori, un solo toolset

Questo è il pivot architetturale dell'intero progetto. **Lo stesso toolset Obsidian-nativo è consumato da due controllori con politiche opposte:**

| | **Loop agentico** (`silica`) | **Pipeline critica** (Injector/…) |
|---|---|---|
| Controllo | LLM in loop, libertà alta | macchina a stati, libertà zero |
| Determinismo | non-deterministico | deterministico, riproducibile |
| Presenza umana | human-in-the-loop | unattended |
| Garanzie | best-effort | gate di accettazione + rollback |
| Esempio | "sistema le note di oggi sulle reti neurali" | `golden_pipeline_run` |

La regola che previene la regressione: **una pipeline a gate non deve mai essere "le sue fasi improvvisate dal modello".** Se l'agente, dovendo ingerire un inbox, ri-decidesse fase per fase quali tool chiamare, la garanzia che "payload >80KB *viene* partizionato" e che "il gate ≥10% *spara*" diventerebbe solo statistica. Per una pipeline a gate, *"di solito partiziona"* è un bug.

Quindi: **l'Injector è una funzione deterministica con i gate cablati, esposta all'agente come una singola azione** (`silica_run_injector`). L'agente *orchestra* (decide *quando* e *su cosa* lanciarla); la pipeline *esegue* in modo blindato. È esattamente il pattern che già vivi in Hermes — il loop è agentico, ma le skill Injector/Refiner sono playbook deterministici che il modello *attiva*. Silica lo rende struttura, non convenzione di prompt.

```
            ┌─────────────────────────────────────────────────┐
            │  Utente: "ingerisci l'inbox di oggi in Deep L."  │
            └───────────────────────┬─────────────────────────┘
                                    ▼
                    ┌───────────────────────────┐
                    │   LOOP AGENTICO (silica)   │   ← libertà alta
                    │   ragiona, sceglie azione  │
                    └───────────┬───────────────┘
                                │ invoca come AZIONE SINGOLA
                                ▼
                    ┌───────────────────────────┐
                    │  PIPELINE INJECTOR         │   ← libertà zero, gate cablati
                    │  recon→payload→delega→     │
                    │  validate→write→lint→clean │
                    └───────────┬───────────────┘
                                │ usano gli STESSI tool atomici
                                ▼
              ┌───────────────────────────────────────┐
              │  TOOLSET OBSIDIAN-NATIVO (L0 + L1)      │
              └───────────────────────────────────────┘
```

---

## 3. Architettura a strati (L0–L4)

Cinque strati, mappati uno-a-uno sulla frase-filosofia. Il keystone è L0.

### L0 — Obsidian Driver (l'astrazione sul substrato I/O)

Adapter tipizzato **per dominio**, non per trasporto. Tutto il resto parla al Driver, mai a disco o CLI direttamente. Due backend intercambiabili dal giorno 1:

- **`cli`** — *backend primario*. Wrappa la CLI ufficiale di Obsidian (richiede l'app desktop ≥1.12.7 in esecuzione: è un bridge CDP all'istanza Electron). Legge la **metadata-cache viva** e il **motore del grafo**: latenza ~zero, write graph-safe (i wikilink vengono aggiornati dal motore di Obsidian).
- **`fs`** — *backend degradato + oracolo di regressione*. Filesystem diretto + indice mantenuto (sqlite/in-memory). Derivato dagli script Hermes attuali (`recon.py`, `find_duplicates.py`, `frontmatter.py`). Serve per l'headless unattended **e** come reference contro cui validare il backend `cli` (vedi §6).

**Perché due implementazioni dal giorno 1:** due implementazioni concrete sono il numero minimo che rende un'astrazione *onesta*. Con un solo backend si finisce con una *leaky abstraction* — metodi che sono CLI travestita (passi flag, parsi stringhe). Avere `fs` come seconda implementazione reale costringe l'interfaccia a esprimersi in termini di dominio.

**Interfaccia (firma di dominio, non di trasporto):**

```python
class ObsidianDriver(Protocol):
    # lettura / scoperta
    def search_names(self, query: str) -> list[NoteRef]: ...
    def search_context(self, query: str) -> list[Hit]: ...        # snippet + righe
    def read_note(self, ref: NoteRef) -> NoteContent: ...
    def props_of(self, ref: NoteRef) -> dict: ...                 # frontmatter, ~centinaia di token
    def outline(self, ref: NoteRef) -> Heading: ...
    # grafo
    def links(self, ref: NoteRef) -> list[NoteRef]: ...
    def backlinks(self, ref: NoteRef) -> list[NoteRef]: ...
    def orphans(self) -> list[NoteRef]: ...
    def unresolved(self) -> list[Link]: ...
    def graph_snapshot(self) -> GraphSnapshot: ...                # per il gate di non-regressione
    # scrittura (graph-safe)
    def create(self, path: str, content: str, template: str | None = None) -> NoteRef: ...
    def append(self, ref: NoteRef, content: str) -> None: ...
    def set_prop(self, ref: NoteRef, name: str, value, type_: str) -> None: ...
    def move(self, ref: NoteRef, to: str) -> None: ...            # aggiorna i wikilink
    def delete(self, ref: NoteRef) -> None: ...
    # avanzate / escape hatch
    def base_query(self, base: str, view: str) -> list[dict]: ... # DB sul frontmatter
    def eval(self, js: str) -> object: ...                        # app-bound, superficie di sicurezza
    # transazionalità
    def snapshot_versions(self, refs: list[NoteRef]) -> Txn: ...  # via history:/sync:
    def restore(self, txn: Txn) -> None: ...                      # rollback su gate fallito
```

> **Contratto di freshness (NORMATIVO).** Il Driver DEVE dichiarare la semantica read-after-write. Il backend `cli` legge la cache viva: dopo una `create`/`set_prop`/`move`, il Driver garantisce che la lettura successiva rifletta la mutazione — se la cache di Obsidian si aggiorna in modo asincrono, è il backend a dover attendere/poll fino a settle (l'astrazione nasconde il dettaglio). Il backend `fs` mantiene l'indice e DEVE invalidarlo a ogni write. Un metodo che non rispetta lo stesso contratto di freshness su entrambi i backend è un bug, non una differenza di implementazione.

### L1 — Kernel meccanico (deterministico, zero LLM)

Funzioni *pure* sugli output del Driver. Testabili in isolamento con golden test. → *"i lavori meccanici devono essere eseguiti da script"*.

- collision detection (l'ex `recon.py`, ridotto al filtro `NOISE_PATTERNS` + composizione di `search_context`);
- partizionamento del payload (`--max-concepts`, soglia 80KB);
- normalizzazione frontmatter (regola lowercase + hyphen dei tag);
- splitting di atomicità (≤40 righe / 6000 char, hub-and-spoke);
- OFM linter (callout, `[[Hub]]` presente, math/mermaid, balance dei delimitatori);
- sanitizzazione output sub-agent (strip fence/preambolo);
- scoring di accettazione (soglia rigetti ≥10% → abort);
- **graph-diff di non-regressione** (diff di due `GraphSnapshot`).

### L2 — Worker semantici (sub-agent, CoT-intensivi, stateless)

Ricevono un prompt renderizzato *verbatim* + un puntatore al payload, restituiscono JSON di ops *stretto*, **non toccano mai il vault**. → *"i lavori CoT-intensivi suddivisi in batch e delegati ai sub-agent"*.

- **Distiller** — confronto concetto-inbox vs nota-vault → decisione `enrich | create | skip` + body markdown.
- **Merger** — unificazione di duplicati senza perdita di densità (pipeline Dedup).
- **Splitter-judge** — confini semantici per l'atomizzazione (pipeline Refiner).

### L3 — Router / Orchestrator (il direttore)

Macchina a stati. Possiede transizioni di fase, gate di accettazione, graph-diff, ledger di idempotenza, policy abort/retry, snapshot/rollback. Chiama L1 per la meccanica, L2 per la semantica, L0 per l'I/O. **Non è un LLM in loop libero** — è codice deterministico che invoca LLM *solo al confine dei worker* (input → structured output bounded). → *"il router orchestrerà l'intera pipeline e verificherà i criteri di accettazione e non-regressione"*.

### L4 — Pipeline come ricette dichiarative

Injector/Refiner/Dedup come grafi di fasi (YAML). Nuova pipeline = nuova ricetta, non nuovo codice orchestratore.

```yaml
# recipes/injector.yaml  (bozza)
name: injector
inputs: [inbox, target, hub]
gates:
  rejection_rate_max: 0.10
  graph_regression: forbid_new_orphans
phases:
  - { id: recon,        kind: mechanical, tool: silica_recon }
  - { id: payload,      kind: mechanical, tool: silica_payload, partition_if_over: 200 }
  - { id: distill,      kind: semantic,   worker: distiller, fanout: true, max_workers: 7 }
  - { id: sanitize,     kind: mechanical, tool: silica_sanitize }
  - { id: validate,     kind: gate,       tool: silica_validate_ops, abort_code: 2 }
  - { id: snapshot,     kind: txn,        tool: silica_snapshot }
  - { id: write,        kind: mechanical, tool: silica_bulk_write }
  - { id: lint,         kind: gate,       tool: silica_lint }
  - { id: cleanup,      kind: mechanical, tool: silica_cleanup, on_success_only: true }
  - { id: rollback,     kind: txn,        tool: silica_restore, on_gate_fail: true }
```

### Mappa strati ↔ filosofia

| Strato | Frase-filosofia | Natura |
|---|---|---|
| L0 Driver | (substrato) | I/O, due backend |
| L1 Kernel | "meccanico → script" | deterministico, no LLM |
| L2 Worker | "CoT → batch → sub-agent" | LLM bounded |
| L3 Router | "orchestra + verifica gate/regressione" | macchina a stati |
| L4 Ricette | (forma delle pipeline) | dichiarativo |

---

## 4. Il catalogo dei tool

### 4.1 Tassonomia

Tre classi. Mischiarle è la fonte principale di regressione.

- **Atomici** — un'operazione Obsidian-nativa, 1:1 su un comando CLI o su una funzione pura del kernel. Vocabolario di base. Li chiama sia l'agente sia la pipeline.
- **Composti** — *un intero script promosso ad azione singola*, con i suoi invarianti dentro. Incapsulano la logica che il modello NON deve riscoprire.
- **Wrapped** — tool di scrittura con le Golden Rule cablate dentro: rendono *impossibile per costruzione* la violazione, invece di chiederla per prompt.

### 4.2 Tool atomici

| Tool | Comando CLI ufficiale sottostante | Note |
|---|---|---|
| `silica_search(q)` | `search query= format=json` | match sui nomi nota |
| `silica_search_context(q)` | `search:context query= format=json` | snippet + righe (era `search-content` di yakitrak — NON esiste nell'ufficiale) |
| `silica_read_note(name)` | `read file=` | risoluzione wikilink-style, niente path guessing |
| `silica_props(name)` | `property:read` / `properties file= format=json` | routing su ~centinaia di token, niente lettura del corpo |
| `silica_set_prop(name, k, v, type)` | `property:set type=` | la *regola* di normalizzazione sta in L1, l'*applicazione* qui |
| `silica_outline(name)` | `outline file= format=json` | albero degli heading |
| `silica_links(name)` | `links file=` | uscenti |
| `silica_backlinks(name)` | `backlinks file= counts` | entranti |
| `silica_orphans()` | `orphans` | per il graph snapshot |
| `silica_unresolved()` | `unresolved` | per il graph snapshot |
| `silica_base_query(base, view)` | `base:query view= format=json` | DB sul frontmatter |
| `silica_eval(js)` | `eval code=` | escape hatch, app-bound, **uso parsimonioso** |

### 4.3 Tool composti (script promossi)

| Tool | Script-sorgente Hermes | Invariante preservato |
|---|---|---|
| `silica_recon(inbox, vault)` | `recon.py` | filtro `NOISE_PATTERNS`; stesso schema JSON di output. **I/O reinstradato sui tool atomici, non più `os.walk`.** |
| `silica_payload(recon, max_concepts)` | `distiller_payload.py` | partizionamento obbligatorio >200 concetti / >80KB |
| `silica_prep_delegation(protocol, payload)` | `prep_delegation.py` | prompt renderizzato *verbatim*; payload by-pointer; checksum SHA-256 |
| `silica_sanitize(raw)` | `parse_distiller_output.py` | strip fence/preambolo; heading non nel payload → reject |
| `silica_validate_ops(ops, payload)` | `validate_operations.py` | **exit code 2 a ≥10% rigetti** → ritorno strutturato che il router tratta come abort |
| `silica_bulk_write(ops)` | `bulk_writer.py` | scrittura sempre via templating coerente, mai write raw |
| `silica_lint(ops, hub)` | `linter.py` + `hermes_common/ofm.py` | atomicità, OFM, `[[Hub]]` presente |
| `silica_find_duplicates(vault)` | `find_duplicates.py` | group-by-basename; I/O su `silica` `files`/`base_query` |
| `silica_run_injector(inbox, target, hub)` | `golden_pipeline_run.md` | l'intera ricetta, gate inclusi — **azione singola per l'agente** |

### 4.4 Tool wrapped sulle Golden Rule

> **Decisione cristallizzata.** Le invarianti vivono nel toolset, non nel system prompt. Un system prompt è una raccomandazione; un tool wrapped è un'invariante.

- `silica_move` aggiorna *sempre* i wikilink (è l'unico modo di spostare: **non esiste** un tool che fa `mv` raw). L'EMOTION PROMPT Hermes su `move` diventa un'invariante del tool.
- `silica_delete` rifiuta — o richiede conferma esplicita umana — se cancellerebbe densità non altrove preservata (anti-deletion policy).
- Ogni write passa da `silica_lint` *prima* del commit: un'operazione che non supera il lint non viene scritta.

### 4.5 Lo schema di un tool (il contratto `@tool`)

Ogni tool dichiara: nome, descrizione (è il prompt che il modello legge), schema dei parametri (validato), e la classe (atomic/composed/wrapped). Vedi §8.4 per l'implementazione del registry.

---

## 5. Le Golden Rule (invarianti di dominio)

Ereditate dal playbook Hermes, ora **enforced nei tool wrapped e nel linter**, non lasciate al prompt:

1. **Anti-deletion** — mai cancellare densità in modo silente; preferire `patch`/`append`/unificazione. Eccezioni ammesse: rumore semantico/formattazione, riscrittura più completa dello stesso concetto, dato verificato errato via web.
2. **Atomicità modulare (hub-and-spoke)** — Spoke ≤40 righe / 6000 char, ciascuno con `[[Hub]]` nel corpo.
3. **OFM compliance** — callout (`> [!tip]`), block ref (`^id`), Mermaid, LaTeX (`$$…$$`).
4. **Densità fattuale** — estrarre definizioni, formule, schemi, esempi; niente riassunti hand-wavy.
5. **Provenienza AI** — frontmatter `ai_generated: true` congelato alla scrittura sulle note generate.
6. **Normalizzazione tag** — lowercase + hyphen (`reti-neurali`, `machine-learning`).
7. **Leggibilità accademica** — italiano formale, keyword in grassetto, struttura per studiosi.
8. **Idempotenza** — file inbox processati spostati in `done/` per-batch; un resume non riprocessa.

---

## 6. Strategia di non-regressione

È il requisito che hai posto per primo. Tre meccanismi, in ordine di forza.

### 6.1 Promozione script → tool (refactor I/O-only)

Gli script attuali **non vengono sostituiti da tool**: vengono *promossi* a tool composti, conservando il comportamento verificato, e **solo lo strato di I/O** viene reinstradato sulla CLI ufficiale. `recon.py` continua a trovare le stesse collisioni, filtrare lo stesso rumore, emettere lo stesso JSON — ma interroga la cache viva invece di camminare il disco. *Stesso contratto di output, sorgente diversa.* È il refactor più sicuro possibile: si tocca solo ciò che non si fida ancora.

### 6.2 Il backend `fs` come oracolo golden

Il backend `fs` (gli script di oggi) è la **reference contro cui validare il backend `cli`**. Golden test: esegui `silica_recon` su backend `fs` e su backend `cli` sullo stesso vault, confronta i due JSON. Se divergono, **hai una regressione misurabile prima di spedire**. Questo lega il doppio-backend (deciso in §3) al requisito di non-regressione: l'fs non è solo degraded-mode, è il giudice.

### 6.3 Suite di regressione sul grafo (gate di accettazione 2.0)

Snapshot di `orphans` + `unresolved` + conteggi `backlinks`/`links` *prima* e *dopo* ogni batch. Il gate diventa un **diff sul grafo**: ho creato orfani? aumentati gli unresolved link? rotto backlink esistenti? La "non-regressione" diventa misurabile a livello di grafo, non solo di file.

### 6.4 Rollback transazionale nativo

`history:restore` / `sync:restore` danno un undo versionato. Il router snapshotta *prima* di un batch (`silica_snapshot`) e, se un gate fallisce, ripristina (`silica_restore`). Risolve l'atomicità di batch che `bulk_writer` non garantiva.

---

## 7. La pipeline Injector — prima cittadina

### 7.1 Perché l'Injector per primo

Per un secondo cervello, **l'ingestione è la value proposition**; Dedup e Refiner sono manutenzione. Costruire Dedup per primo avrebbe validato l'architettura senza validare il prodotto. In più, l'Injector è l'unica pipeline che esercita l'arco completo (recon→payload→delega→validate→write→lint→cleanup): se sopravvive lui, gli altri due sono sottoinsiemi. Ed è già specificato in modo deterministico in `golden_pipeline_run.md` — non lo si scopre, lo si *porta* su L0/L1.

### 7.2 MVP = walking skeleton, non l'Injector completo

Una **fetta verticale** che attraversa tutti e 5 gli strati su scope ridotto:

- inbox a **file singolo**, un solo `TARGET`, `--max-concepts` basso;
- **un solo batch** — niente fan-out parallelo ancora;
- ma con il **gate ≥10% cablato** e il **rollback via `history:restore`** funzionante.

Un concetto che entra end-to-end con il gate che spara davvero. La complessità vera (partizionamento >200, `ThreadPoolExecutor` a max concorrenza, merge dei batch) si aggiunge *dopo* che lo scheletro cammina.

### 7.3 Fasi e gate (dal golden run)

```
Phase 1  recon          → /tmp/recon.json            (mechanical)
Phase 2.0 payload       → /tmp/distiller_payload.json (mechanical, partition se >200/>80KB)
Phase 2.1 delegate      → distiller sub-agent(s)      (semantic, fanout opzionale)
Phase 2.2 sanitize      → ops.json                    (mechanical)
Phase 2.3 validate      → exit 0 ok | exit 2 ABORT     (GATE: rigetti ≥10%)
Phase 2.5 snapshot      → Txn                          (txn, pre-write)
Phase 3  bulk_write     → muta il vault                (mechanical)
Phase 4  lint           → OFM/atomicità/hub            (GATE)
         graph-diff     → no nuovi orfani/unresolved   (GATE)
Phase 5  cleanup        → inbox → done/  (solo se tutti i gate verdi) | restore (se rosso)
```

Hard-stop (ereditati): >200 concetti → partizione obbligatoria; payload >80KB → partizione; fan-out > max concorrenza (7, max 10) → errore esplicito; router context > 60k token → stop.

---

## 8. Come si instanzia una CLI agentica — bootstrap pratico

Sezione per chi non ha mai costruito un agentic-framework. L'obiettivo è **non reinventare la ruota**: prendere il pattern minimo da Hermes (Python, `uv`, entry point `[project.scripts]`, provider-LLM astratto, TUI `prompt_toolkit`) e nient'altro.

### 8.1 L'anatomia di un loop agentico (cosa "è" davvero)

Spogliato di tutto, un agente è **un `while` attorno a una chiamata LLM con function-calling**:

```
loop:
  risposta = LLM(system_prompt, storia_messaggi, schemi_dei_tool)
  se risposta contiene tool_calls:
      per ogni tool_call:
          risultato = esegui_tool(nome, argomenti)
          appendi alla storia un messaggio "tool_result"
      continua            # ri-chiama l'LLM con i risultati
  altrimenti:
      mostra risposta.testo all'utente
      attendi prossimo input utente
```

Tutto il resto (streaming, TUI, compressione contesto, multi-provider) è ergonomia attorno a questo nucleo. **Costruisci prima questo nucleo, poi l'ergonomia.**

### 8.2 Stack tecnologico raccomandato (inferito da Hermes)

- **Python 3.11+**, ambiente e packaging con **`uv`**.
- **`litellm`** per l'astrazione provider-LLM (OpenRouter/Anthropic/OpenAI/locale, function-calling uniforme — è il "use any model, no lock-in" di Hermes senza scriverlo tu).
- **`pydantic`** per gli schemi dei tool (validazione + generazione JSON-schema automatica).
- **`prompt_toolkit`** per la TUI (multiline, autocomplete slash-command, history).
- **`concurrent.futures.ThreadPoolExecutor`** (stdlib) per il fan-out dei sub-agent.
- **`sqlite3`** (stdlib) per ledger di idempotenza e log.
- entry point CLI via `pyproject.toml → [project.scripts]`.

### 8.3 Struttura della repo

```
silica/
├── pyproject.toml              # [project.scripts] silica = "silica.cli:main"
├── SILICA.md                   # questo documento
├── silica/
│   ├── cli.py                  # entry point: REPL TUI → loop agentico
│   ├── agent/
│   │   ├── loop.py             # il while agentico (§8.4)
│   │   ├── llm.py              # wrapper litellm provider-agnostico
│   │   └── delegate.py         # fan-out sub-agent (ThreadPoolExecutor)
│   ├── tools/
│   │   ├── registry.py         # @tool decorator + TOOLS dict + JSON schema
│   │   ├── atomic.py           # silica_search, silica_read_note, ... (facciate L0)
│   │   ├── composed.py         # silica_recon, silica_validate_ops, ... (script promossi)
│   │   └── wrapped.py          # silica_move/delete/write con Golden Rule
│   ├── driver/
│   │   ├── base.py             # ObsidianDriver Protocol + contratto freshness
│   │   ├── cli_backend.py      # shell-out su `obsidian ... format=json`
│   │   └── fs_backend.py       # filesystem + indice (deriva da recon.py/frontmatter.py)
│   ├── kernel/                 # L1: funzioni pure (ex hermes_common + script logic)
│   │   ├── frontmatter.py      # AS-IS da Hermes
│   │   ├── ofm.py              # AS-IS da Hermes
│   │   ├── partition.py
│   │   ├── sanitize.py
│   │   ├── accept.py           # scoring gate ≥10%
│   │   └── graphdiff.py        # non-regressione sul grafo
│   ├── router/
│   │   └── orchestrator.py     # L3: macchina a stati che esegue le ricette
│   ├── recipes/
│   │   ├── injector.yaml       # L4
│   │   ├── refiner.yaml
│   │   └── dedup.yaml
│   ├── workers/
│   │   ├── distiller.py        # L2 prompt + dispatch
│   │   └── prompts/
│   │       └── distiller_prompt.txt
│   └── cron/
│       └── schedule.py         # "cura mentre dormo" + report sink monodirezionale
└── tests/
    └── golden/                 # fs-vs-cli regression oracle (§6.2)
```

### 8.4 Lo scheletro minimo (codice)

**Tool registry** — `silica/tools/registry.py`:

```python
import inspect, functools
from typing import Callable
from pydantic import BaseModel

class Tool:
    def __init__(self, fn: Callable, name: str, description: str,
                 params_model: type[BaseModel], cls: str):
        self.fn, self.name, self.description = fn, name, description
        self.params_model, self.cls = params_model, cls  # cls: atomic|composed|wrapped
    def json_schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.params_model.model_json_schema()}
    def run(self, **kwargs):
        return self.fn(**self.params_model(**kwargs).model_dump())

TOOLS: dict[str, Tool] = {}

def tool(params_model: type[BaseModel], cls: str = "atomic"):
    def deco(fn):
        TOOLS[fn.__name__] = Tool(fn, fn.__name__, fn.__doc__ or "", params_model, cls)
        return fn
    return deco
```

**Definizione di un tool atomico** — `silica/tools/atomic.py`:

```python
from pydantic import BaseModel
from .registry import tool
from ..driver import DRIVER  # istanza scelta a runtime: cli (default) o fs

class ReadNoteArgs(BaseModel):
    name: str

@tool(ReadNoteArgs, cls="atomic")
def silica_read_note(name: str):
    """Legge una nota del vault per nome (risoluzione wikilink-style). NON usare path."""
    return DRIVER.read_note(name)
```

**Il loop agentico** — `silica/agent/loop.py`:

```python
from .llm import call_llm
from ..tools.registry import TOOLS

def run_agent(messages: list[dict], model: str) -> str:
    schemas = [t.json_schema() for t in TOOLS.values()]
    while True:
        resp = call_llm(model, messages, tools=schemas)   # litellm dietro
        messages.append(resp.assistant_message)
        if not resp.tool_calls:
            return resp.text                                # risposta finale
        for tc in resp.tool_calls:
            try:
                result = TOOLS[tc.name].run(**tc.args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": str(result)})
        # loop: ri-chiama l'LLM con i risultati dei tool
```

**Fan-out sub-agent** — `silica/agent/delegate.py`:

```python
from concurrent.futures import ThreadPoolExecutor

def delegate(tasks: list[dict], run_one, max_workers: int = 7):
    """tasks = lista di payload renderizzati; run_one(task) -> output grezzo.
    Hard-stop: se len(tasks) > 10 sollevare, non troncare."""
    if len(tasks) > 10:
        raise RuntimeError(f"fan-out {len(tasks)} > max 10: ripartizionare")
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as ex:
        return list(ex.map(run_one, tasks))
```

**Entry point CLI** — `silica/cli.py`:

```python
from prompt_toolkit import PromptSession
from .agent.loop import run_agent

def main():
    session = PromptSession()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("silica — agente Obsidian. /exit per uscire.")
    while True:
        try:
            user = session.prompt("silica> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user.strip() in ("/exit", "/quit"):
            break
        messages.append({"role": "user", "content": user})
        answer = run_agent(messages, model=CONFIG.model)
        print(answer)
```

**Registrazione del comando** — `pyproject.toml`:

```toml
[project]
name = "silica"
requires-python = ">=3.11"
dependencies = ["litellm", "pydantic", "prompt_toolkit", "pyyaml"]

[project.scripts]
silica = "silica.cli:main"
```

Dopo `uv pip install -e .`, il comando `silica` è nel PATH. **Questo è l'intero "framework" minimo** — ~150 righe. Tutto il resto del progetto è il toolset Obsidian-nativo (L0/L1) e il router (L3) che ci stanno sopra.

### 8.5 Da Hermes: cosa vendorizzare, cosa ignorare

- **Vendorizza concettualmente** (riscrivi il piccolo pezzo, non importare il monorepo): il pattern di fan-out sub-agent (`delegate`), il render-verbatim del prompt + payload-by-pointer + checksum, la tabella di hard-stop.
- **Riusa AS-IS** (copia i file): `hermes_common/frontmatter.py`, `hermes_common/ofm.py`, `templates.py`.
- **Ignora**: `acp_adapter`, `gateway`, `docker`, backend di esecuzione remoti, `trajectory_compressor`, honcho, messaggistica bidirezionale.
- **Tieni di Hermes, ma minimale**: lo scheduler `cron` + un report sink monodirezionale (anche solo una nota di report scritta nel vault, o un webhook) — servono per "cura mentre dormo", non sono chat.

---

## 9. Eredità da Hermes — sintesi

| AS-IS (libreria meccanica pura) | DA ADATTARE (pattern, non codice) | DA ABBANDONARE |
|---|---|---|
| `ofm.py`, `frontmatter.py`, `templates.py` | Tool Allocation table → contratto fra strati | dipendenza `execute_code`/`delegate_task`/`read_file` |
| gate a soglia (exit 2 ≥10%) | `prep_delegation` (verbatim + pointer + SHA-256) | `recon.py` `os.walk` (→ driver) |
| pattern idempotenza `done/` | EMOTION PROMPT → invarianti/assert nel router e nei tool wrapped | `find_duplicates` walk (→ driver) |
| `golden_pipeline_run.md` come fixture | hard-stops table | yakitrak `obsidian-cli` (interamente) |

---

## 10. Roadmap di implementazione

- **Fase 0 — Scaffold.** Repo §8.3, `pyproject.toml`, loop agentico minimo (§8.4) con UN tool atomico (`silica_read_note`) funzionante end-to-end contro un vault reale. Verifica che `silica` parli con Obsidian-CLI.
- **Fase 1 — Driver + kernel.** `ObsidianDriver` (Protocol + contratto freshness), backend `cli` completo, backend `fs` derivato dagli script. Porta `ofm.py`/`frontmatter.py` AS-IS. Golden test fs-vs-cli su `silica_recon`.
- **Fase 2 — Walking skeleton Injector.** Fetta verticale §7.2: file singolo, un batch, gate ≥10% + rollback `history:restore` cablati. Esposto come `silica_run_injector`.
- **Fase 3 — Injector completo.** Partizionamento >200, fan-out `ThreadPoolExecutor`, merge batch, graph-diff gate.
- **Fase 4 — Refiner + Dedup** come ricette L4 (riuso del router e dei tool).
- **Fase 5 — Cron + report sink.** "Cura mentre dormo": audit notturno, report monodirezionale.

---

## 11. Decisioni cristallizzate (glossario)

- **Backend primario:** `cli` (Obsidian-CLI ufficiale). `fs` costruito dal giorno 1 come degraded-mode + oracolo di regressione.
- **Forma di Silica:** app standalone con loop agentico, toolset Obsidian-nativo. NON framework generale.
- **Router:** macchina a stati deterministica; LLM solo al confine dei worker semantici.
- **Pipeline critiche:** tool composti deterministici con gate cablati, invocati dall'agente come azione singola.
- **Invarianti (Golden Rule):** vivono nei tool wrapped e nel linter, non nel prompt.
- **Prima pipeline:** Injector, come walking skeleton verticale.
- **Tensione headless ↔ app-bound:** *(aperto)* — risolta a v2 promuovendo il backend `fs` a primo cittadino dell'esecuzione unattended, o eseguendo Obsidian sotto xvfb. Da decidere quando il walking skeleton cammina.
