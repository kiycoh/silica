from unittest.mock import patch, MagicMock
from silica.cli import main
from silica.config import CONFIG
from silica.prompts import SYSTEM_PROMPT

def test_cli_clear_command():
    # We will mock prompt returns:
    # 1. "/clear" -> triggers clear and session recreation
    # 2. EOFError -> exits the main loop
    
    mock_session_1 = MagicMock()
    mock_session_2 = MagicMock()
    
    # build_session should return mock_session_1 on the first call (initial startup),
    # and mock_session_2 on the second call (after /clear is processed).
    mock_build_session = MagicMock(side_effect=[mock_session_1, mock_session_2])
    
    # Setup prompt calls
    mock_session_1.prompt.return_value = "/clear"
    mock_session_2.prompt.side_effect = EOFError()  # exit on second loop iteration
    
    # Mock CONFIG.model and CONFIG.vault_name for deterministic output testing
    with patch("silica.cli.build_session", mock_build_session), \
         patch("silica.cli.CONSOLE") as mock_console, \
         patch("silica.cli.print_home") as mock_home, \
         patch("silica.cli._setup_logging"), \
         patch("sys.argv", ["silica"]):

        main()

        # Verify initial home screen and setup calls
        assert mock_home.call_count == 2  # Initial + after clear
        assert mock_console.clear.call_count == 1
        assert mock_build_session.call_count == 2
        mock_session_1.prompt.assert_called_once()
        mock_session_2.prompt.assert_called_once()
