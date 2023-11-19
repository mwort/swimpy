

# ON LOCAL

# make sure tests/project is working
# make sure nothing unintended is copied from swimpy/*

# build image
docker build -t mwort/swim-dashboard -f Dockerfile_swim_dashboard .

# push to docker Hub
docker push mwort/swim-dashboard

# ON SERVER as root
su root
n=12
container_id_file="swim_dashboard_containers.txt"
random_strings_file="random_strings.txt"

docker pull mwort/swim-dashboard

# stop containers
docker stop -t 2 $(cat $container_id_file)

# start containers
rm -f $container_id_file
for i in $(seq 1 $n); do
    url=/swim-dashboard/$(head -n $i $random_strings_file | tail -1)/
    docker run -d -e DASHBOARD_BASE_URL=$url --pids-limit=-1 --rm --name swim-dashboard-$i -p $((8050+$i)):8054 mwort/swim-dashboard:latest >> $container_id_file
    echo https://deltaclimateservices.com$url
done


# then configure proxy rules to the echo'ed URLs