def commit_callback(commit):
    if commit.author_name == b"Pio Dinesh" and commit.author_email == b"piodinesh4@gmail.com":
        commit.author_name = b"Pio Dinesh J"
        commit.author_email = b"dinesh@infinitycorp.tech"
    if commit.committer_name == b"Pio Dinesh" and commit.committer_email == b"piodinesh4@gmail.com":
        commit.committer_name = b"Pio Dinesh J"
        commit.committer_email = b"dinesh@infinitycorp.tech"
