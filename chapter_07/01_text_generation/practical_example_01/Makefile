# Makefile

.SHELL := /bin/bash
.ONESHELL:

.PHONY: run install

run: 
	reset
	. .venv/bin/activate && python3 __main__.py

install: 
	reset
	rm -rf .venv
	python3 -m venv .venv
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt
