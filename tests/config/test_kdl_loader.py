from __future__ import annotations

from config.formats.kdl_loader import loads_kdl


def test_kdl_parses_config_uses_and_nested_values() -> None:
    data = loads_kdl(
        '''
        config target="train" version=2 {
          use "presets/model/mmdit_576.kdl"
          model {
            use "mmdit_576"
            family "mmdit"
            architecture {
              depth 24
              enabled true
              values [1, 2, "x"]
            }
          }
        }
        ''',
        source="<test>",
    )

    assert data["__kind__"] == "config"
    assert data["__meta__"] == {"target": "train", "version": 2}
    assert data["__uses__"] == ["presets/model/mmdit_576.kdl"]
    assert data["model"]["__uses__"] == ["mmdit_576"]
    assert data["model"]["family"] == "mmdit"
    assert data["model"]["architecture"]["depth"] == 24
    assert data["model"]["architecture"]["enabled"] is True
    assert data["model"]["architecture"]["values"] == [1, 2, "x"]
