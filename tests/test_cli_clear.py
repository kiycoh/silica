from unittest.mock import patch, MagicMock
from silica.cli import main


def test_cli_clear_command():
    # /clear resets the conversation in place via _fresh_messages() — it does NOT
    # rebuild the prompt session. One session serves both loop iterations:
    #   1. "/clear"  -> clear console, reprint home, reset messages, continue
    #   2. EOFError  -> exit the main loop
    mock_session = MagicMock()
    mock_session.prompt.side_effect = ["/clear", EOFError()]
    mock_build_session = MagicMock(return_value=mock_session)

    with patch("silica.cli.build_session", mock_build_session), \
         patch("silica.cli.CONSOLE") as mock_console, \
         patch("silica.cli.print_home") as mock_home, \
         patch("silica.cli._setup_logging"), \
         patch("silica.cli._update_context_tokens") as mock_update, \
         patch("sys.argv", ["silica"]):

        main()

        # Session is built once at startup and reused across /clear (no rebuild).
        assert mock_build_session.call_count == 1
        # /clear clears the console once; home is printed at startup + on /clear.
        assert mock_console.clear.call_count == 1
        assert mock_home.call_count == 2
        # _fresh_messages() recomputes the token count at startup and on /clear.
        assert mock_update.call_count == 2
        # Two prompts consumed: the "/clear" command, then the EOF that exits.
        assert mock_session.prompt.call_count == 2
