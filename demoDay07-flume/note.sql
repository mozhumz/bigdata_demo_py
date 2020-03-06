create table weblogs (
    id string,
    msg string
)
clustered by (id) into 5 buckets
stored as orc;