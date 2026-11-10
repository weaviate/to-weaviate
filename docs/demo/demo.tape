# Records docs/demo/demo.gif, the README demo: inspect, the start of the
# plan, check, copy and verify.
#
# Needs Chroma on port 8001 and Weaviate on ports 8099 and 50099 with the API
# key admin-key (`make chroma-up` and `make weaviate-up`), and to-weaviate on
# PATH. From the repository root:
#
#   python docs/demo/seed.py
#   PATH="$PWD/.venv/bin:$PATH" vhs docs/demo/demo.tape
#
# The hidden part deletes the Weaviate collection Support_articles and moves
# to a new temporary directory, so every recording starts a new migration.
# The screen is cleared, hidden, between the steps.

Output docs/demo/demo.gif

Require to-weaviate
Require curl

Set Shell bash
Set FontSize 18
Set Width 1200
Set Height 620
Set Padding 24
Set WindowBar Colorful
Set BorderRadius 8
Set TypingSpeed 40ms

Hide
Type "export PS1='$ ' CHROMA_URL=http://localhost:8001 WEAVIATE_URL=http://localhost:8099 WEAVIATE_GRPC_URL=http://localhost:50099 WEAVIATE_API_KEY=admin-key"
Enter
Type "curl -s -o /dev/null -X DELETE -H 'Authorization: Bearer admin-key' http://localhost:8099/v1/schema/Support_articles"
Enter
Type "cd $(mktemp -d) && clear"
Enter
Wait@10s /^\$\s*$/
Show

Sleep 500ms
Type "to-weaviate inspect chroma --collection support-articles"
Sleep 300ms
Enter
Wait@30s /^\$\s*$/
Sleep 3s

Hide
Type "clear"
Enter
Wait@10s /^\$\s*$/
Show

Type "head -n 22 .to-weaviate/plan.yaml"
Sleep 300ms
Enter
Wait@10s /^\$\s*$/
Sleep 4s

Hide
Type "clear"
Enter
Wait@10s /^\$\s*$/
Show

Type "to-weaviate check"
Sleep 300ms
Enter
Wait@30s /^\$\s*$/
Sleep 1s
Type "to-weaviate copy"
Sleep 300ms
Enter
Wait@60s /^\$\s*$/
Sleep 1s
Type "to-weaviate verify"
Sleep 300ms
Enter
Wait@30s /^\$\s*$/
Sleep 3500ms
