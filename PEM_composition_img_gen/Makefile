##############################
# Global setup
.ONESHELL: # Source: https://stackoverflow.com/a/30590240
.SILENT: # https://stackoverflow.com/a/11015111

include .env
export $(shell sed 's/=.*//' .env)  # TODO: this fails if the vars are quoted; I think comments lead to error too

##############################
# Functions
# If a function may be executed inside `exec_cluster`, then it must be written in a single line
# Otherwise it may be "normal"
define exec_docker
	# Run a command inside a running container
	# Arguments:
	#  $(1): Command to run
	# Returns:
	#  Output of the command
	docker-compose exec notebooks $(1)
endef

define is_docker_running
	# Check if the docker container is running
	# Arguments
	#  $(1): Container name
	# Returns:
	#  Exit code 0 if running, 1 if not running
	docker-compose ps --filter "status=running" | grep $(1) | grep -v Exit >/dev/null
endef

define exec_cluster
	# Run a command inside the master/base/login node of the cluster
	# Arguments:
	#  $(1): Command to run; String; Should be one single command (chain with `;` or `&&` if needed); Multiline commands need to be escaped with backsplash
	# Returns:
	#  Output of the command
	if [ -z "${SSH_USER_SHIBBOLETH}" ]; then \
		echo "Error: SSH_USER_SHIBBOLETH is not set or is empty"; \
		exit 1; \
	fi;
	sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -o ProxyCommand="sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" ${SSH_USER_CLUSTER}@cl.itk.ppke.hu $(1)
endef

define build_cluster
	`# Generates the .sif file from the .def file`\
	`# Can be executed in ssh`\
	cd '${GIT_REPOSITORY_NAME}/PEM_composition_img_gen'; \
	if [ ! -f .env ]; then \
		echo -e 'You must configure the secrets with a local .env file. Exiting...'; \
		exit 1; \
	fi; \
	if [ ! -f run_cluster.sif ]; then \
		echo -e 'Building run_cluster.sif...'; \
		apptainer build --disable-cache run_cluster.sif run_cluster.def; \
	else \
		echo -e 'run_cluster.sif already exists, skipping build...'; \
	fi;
endef

define check_gpu
	# Check the number of available GPUs
	# Arguments:
	#  $(1): GPU ID
	# Returns:
	#  Number of processes running on the GPU
    nvidia-smi -i $(1) --query-compute-apps=pid --format=csv,noheader | wc -l
endef

##############################
# Targets for docker container
# Mainly for local development
# Only interactive mode is supported (but potentially would make sense to have batch mode too)
# TODO: push the image to dockerhub
run-interactive-docker:
	echo '\n-------------------------------------------------------------------------\n'
	echo 'Go to http://localhost:8888/tree?token=20e10069-1b7c-403f-9895-f6650f4e90d9'
	echo '\n-------------------------------------------------------------------------\n'
	if $(call is_docker_running,unlearning-notebooks); then \
		echo 'Container is already running.'
	else
		echo 'Container is not running. Starting it now...'
		docker-compose down
		docker-compose up -d
		$(call exec_docker, sh -c 'huggingface-cli login --token ${HF_TOKEN}')
		$(call exec_docker, accelerate config default)
	fi;

clean-docker:
	docker-compose down --rmi all
	make run-docker

stop-docker:
	docker-compose down

##############################
# Targets for remote slurm cluster, both batch jobs and interactive jobs
# Docs: https://ppke.sharepoint.com/sites/itk-it/SitePages/HPC.aspx
# All executions are done with apptainer, so you can debug it locally if needed, for example:
# apptainer exec run_cluster.sif python ./fusanalysis/users/Leonardo/3.1_train_cluster.py
# Apptainer has a volume-like mount, so filesystem modifications inside the running "container" are sync with the master node filesystem

setup-cluster:
	# TODO: automate the initial clone of the repo and copy of the .env file
	# For now, you can to do this yourself
	echo 'NOT IMPLEMENTED...'
	exit 1


run-interactive-cluster:
	if $(call is_docker_running,unlearning-notebooks); then \
		echo 'Local docker container can not be running at the same time (there will be port conflict). Exiting...';
		exit 1;
	fi;

	$(call exec_cluster, "$(build_cluster)")

	# TODO: for some reason, the following commands don't work inside the ssh function
	# Use `--gres` to select machine config (gpu:v100:1 = 1 V100 16GB GPU, gpu:a100:1 = 1 A100 40GB GPU)
	sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -o ProxyCommand="sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" ${SSH_USER_CLUSTER}@cl.itk.ppke.hu " \
		cd '${GIT_REPOSITORY_NAME}'; \
		nohup \
		srun -pgpu --gres=gpu:a100:1 apptainer run --nv PEM_composition_img_gen/run_cluster.sif /usr/bin/tini -s -- jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=cfbc4b9c-7056-4a8c-8c34-3e521dd01cdb \
		> output.log 2>&1 & \
	"
	echo 'Job launched'
	sleep 3

	JOB_ID=$$( \
		sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -o ProxyCommand="sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" ${SSH_USER_CLUSTER}@cl.itk.ppke.hu "\
			sacct -u \$$USER --format=JobID,State,Start -n | sort -k3 -r | head -n 1 | awk '{print \$$1}' | sed 's/\..*\$$//' \
		"\
	)
	echo "Latest Job ID: $$JOB_ID"

	
	export NODE=$$( \
		sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -o ProxyCommand="sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" ${SSH_USER_CLUSTER}@cl.itk.ppke.hu "\
			scontrol show job $$JOB_ID | grep -oP '(?<=NodeList=)\S+' | grep -v '(null)'\
		"\
	)
	echo "Node: $$NODE"

	echo "\n-------------------------------------------------------------------------\n"
	echo "Go to http://localhost:8888/tree?token=cfbc4b9c-7056-4a8c-8c34-3e521dd01cdb"
	echo "\n-------------------------------------------------------------------------\n"

	echo 'Starting the ssh tunnel...'
	sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -o ProxyCommand="sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" ${SSH_USER_CLUSTER}@cl.itk.ppke.hu -NTL 8888:$$NODE:8888


run-batch-cluster:
	$(call exec_cluster, $(build_cluster))
	$(call exec_cluster, "\
		cd '${GIT_REPOSITORY_NAME}/PEM_composition_img_gen'; \
		sbatch run_batch_cluster.sh \
	")

clean-cluster:
	$(call exec_cluster, "\
		cd '${GIT_REPOSITORY_NAME}/PEM_composition_img_gen'; \
		rm -rf run_cluster.sif; \
		apptainer cache clean -f; \
	")

stop-cluster:
	# Stop all jobs from your user
	# For interactive jobs, just `Control-C` stops the tunnel, but the slurm job is still running
	$(call exec_cluster, "\
		scancel -u \$$USER; \
		echo -e 'All jobs stopped.' \
	")

debug-last-cluster:
	$(call exec_cluster, "\
		cd '${GIT_REPOSITORY_NAME}/PEM_composition_img_gen'; \
		JOB_ID=\$$(sacct -u \$$USER --format=JobID,State,Start -n | sort -k3 -r | head -n 1 | awk '{print \$$1}' | sed 's/\..*\$$//'); \
		echo -e 'Latest Job ID: \$$JOB_ID'; \
		echo -e '\n\\n-------\nJob details using sacct\n-------'; \
		sacct -j \$$JOB_ID; \
		\
		if [ -f assets/cluster_jobs/\$${JOB_ID}-train-err ]; then \
			echo -e '\n\n-------\nJob error\n-------'; \
			cat assets/cluster_jobs/\$${JOB_ID}-train-err; \
		else \
			echo -e '\n\n-------\nJob error file does not exist\n-------'; \
		fi; \
		\
		if [ -f assets/cluster_jobs/\$${JOB_ID}-train-out ]; then \
			echo -e '\n\n-------\nJob output\n-------'; \
			cat assets/cluster_jobs/\$${JOB_ID}-train-out; \
		else \
			echo -e '\n\n-------\nJob output file does not exist\n-------'; \
		fi; \
		echo -e '\n\n-------\nJob output\n-------'; \
	")

##############################
# Targets for testing
# Currently all tests run inside the docker container, but that's just because of the dependencies
test: run-docker
	echo '\n\n------------------------\nMypy Check\n------------------------'
	$(call exec_docker, poetry run mypy --install-types --non-interactive > /dev/null 2>&1)  # hidden output
	$(call exec_docker, poetry run mypy --no-warn-incomplete-stub --disable-error-code import-untyped --explicit-package-bases /libs)

	echo '\n\n------------------------\nPycodestyle Check\n------------------------'
	$(call exec_docker, poetry run pycodestyle --max-line-length=200 --ignore=E701 /libs)

	echo '\n\n-------\nPytest checks\n-------'
	$(call exec_docker, poetry run pytest /tests)
	# Manual tests (requires things like connecting some hardware or doing something interactive)
	# poetry run pytest /tests/**/manual_*.py
	# poetry run pytest --capture=no -k "test_example" /tests/**/manual_example.py
