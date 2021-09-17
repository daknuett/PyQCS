message(meson.source_root())

python.install_sources(
  '__init__.py', 'measurement.py', 'utils.py',
  pure:true,
  subdir:'pyqcs'
)
python.install_sources(
  'state/abc.py',  'state/__init__.py',  'state/pretty_state.py',  'state/state.py',
  pure:true,
  subdir:'pyqcs/state'
)
python.install_sources(
  'util/bytecode.py',  'util/flatten.py',  'util/from_lists.py',  'util/__init__.py',  'util/random_circuits.py',  'util/random_graphs.py',  'util/to_circuit.py',  'util/to_diagram.py',
  pure:true,
  subdir:'pyqcs/util'
)
python.install_sources(
  'gates/abc.py',  'gates/builtins.py',  'gates/circuits.py',  'gates/exceptions.py',  'gates/executor.py',  'gates/gate.py',  'gates/__init__.py',
  pure:true,
  subdir:'pyqcs/gates'
)
python.install_sources(
  'graph/abc.py',  'graph/c_l.py',  'graph/gate.py',  'graph/__init__.py',  'graph/state.py',  'graph/util.py',
  pure:true,
  subdir:'pyqcs/graph'
)
python.install_sources('gates/implementations/__init__.py', pure:true, subdir:'pyqcs/gates/implementations')
