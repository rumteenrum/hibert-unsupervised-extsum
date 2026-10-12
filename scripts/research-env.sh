#!/usr/bin/env bash
# Source from any directory: source /path/to/redundancy-extractive/scripts/research-env.sh
# Expects ROUGE-1.5.5 in a sibling ../tools/ directory; paths may contain spaces.
RESEARCH_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$RESEARCH_PROJECT_ROOT/.venv/bin/activate" ]]; then
  echo 'Create the research .venv and install requirements first.' >&2
  return 1 2>/dev/null || exit 1
fi
source "$RESEARCH_PROJECT_ROOT/.venv/bin/activate"
export ROUGE_HOME="$(cd -- "$RESEARCH_PROJECT_ROOT/../tools/ROUGE-1.5.5" && pwd)"
if [[ -d "$HOME/perl5/lib/perl5" ]]; then
  export PERL5LIB="$HOME/perl5/lib/perl5${PERL5LIB:+:$PERL5LIB}"
fi
unset RESEARCH_PROJECT_ROOT
