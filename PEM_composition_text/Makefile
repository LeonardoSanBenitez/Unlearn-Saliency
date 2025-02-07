.ONESHELL: # Source: https://stackoverflow.com/a/30590240
.SILENT: # https://stackoverflow.com/a/11015111

run-local:
	echo "\n-------------------------------------------------------------------------\n"
	echo "Go to http://localhost:8888/tree?token=20e10069-1b7c-403f-9895-f6650f4e90d9"
	echo "\n-------------------------------------------------------------------------\n"
	if docker-compose ps --filter "status=running" | grep unlearning-notebooks | grep -v Exit >/dev/null; then \
		echo "Container is already running."; \
	else \
		echo "Container is not running. Starting it now..."; \
		docker-compose down; \
		docker-compose up -d; \
	fi;

run-cluster:
	# pass

run-cluster-clean:
	# pass

stop-local:
	docker-compose down

stop-cluster:
	# pass

debug-last-cluster:
	# pass

test: run-local
	# pass