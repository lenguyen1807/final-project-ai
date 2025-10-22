EMAIL="lenguyen18072003@gmail.com"
NAME="lenguyen1807" # nhớ thay đổi name và mail thành github của mình

.phony: commit
commit:
	git -c user.name=${NAME} -c user.email=${EMAIL} commit -m $(MSG)

.phony: push
push:
	git -c user.name=${NAME} -c user.email=${EMAIL} push -u origin main