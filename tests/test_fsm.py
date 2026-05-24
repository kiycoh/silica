from silica.router.orchestrator import InjectorFSM
from silica.tools.registry import TOOLS

def test_injector_fsm_initialization():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm.state.name == "INIT"

def test_silica_run_injector_is_registered():
    assert "silica_run_injector" in TOOLS
    tool = TOOLS["silica_run_injector"]
    assert tool.cls == "composed"
