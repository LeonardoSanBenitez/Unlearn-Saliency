.ONESHELL: # Source: https://stackoverflow.com/a/30590240
.SILENT: # https://stackoverflow.com/a/11015111

run-local:
	echo "\n-------------------------------------------------------------------------\n"
	echo "Go to http://localhost:8888/tree?token=20e10069-1b7c-403f-9895-f6650f4e90d9"
	echo "\n-------------------------------------------------------------------------\n"
	if docker-compose ps --filter "status=running" | grep unlearning-notebooks | grep -v Exit >/dev/null; then
		echo "Container is already running."
	else
		echo "Container is not running. Starting it now..."
		docker-compose down
		docker-compose up -d
		docker-compose exec notebooks sh -c 'huggingface-cli login --token ${HF_TOKEN}'
		docker-compose exec notebooks accelerate config default
	fi;

run-local-clean:
	docker-compose down --rmi all
	make run-local

run-cluster:
	# TODO: automate the ssh connection, so this command can be run from any machine
	# Take the ssh credentials from the .env file
	if [ ! -f .env ]; then \
		echo -e "You must configure the secrets with a local .env file. Exiting..."; \
		exit 1; \
	fi;
	if [ ! -f run_cluster.sif ]; then \
		echo -e "Building run_cluster.sif..."; \
		apptainer build run_cluster.sif run_cluster.def; \
	else \
		echo -e "run_cluster.sif already exists, skipping build..."; \
	fi;

	# Run locally for debug:
	#apptainer exec run_cluster.sif python ./fusanalysis/users/Leonardo/3.1_train_cluster.py
	
	sbatch run_cluster.sh

run-cluster-clean:
	rm -rf run_cluster.sif
	apptainer cache clean -f
	make run-cluster

stop-local:
	docker-compose down

stop-cluster:
	scancel -u $$USER

debug-last-cluster:
	JOB_ID=$$(sacct -u $$USER --format=JobID,State,Start -n | sort -k3 -r | head -n 1 | awk '{print $$1}' | sed 's/\..*$$//')
	echo -e "Latest Job ID: $$JOB_ID"
	echo -e "\n\n-------\nJob details using sacct\n-------"
	sacct -j $$JOB_ID
	echo -e "\n\n-------\nJob error\n-------"
	cat assets/cluster_jobs/$${JOB_ID}-train-err
	echo -e "\n\n-------\nJob output\n-------"
	cat assets/cluster_jobs/$${JOB_ID}-train-out

test: run-local
	# pass