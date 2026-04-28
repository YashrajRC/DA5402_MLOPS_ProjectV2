#!/bin/sh
sed \
  -e "s/\${MAILTRAP_USER}/$MAILTRAP_USER/g" \
  -e "s/\${MAILTRAP_PASS}/$MAILTRAP_PASS/g" \
  -e "s/\${ALERT_EMAIL}/$ALERT_EMAIL/g" \
  /etc/alertmanager/alertmanager.yml > /tmp/alertmanager.yml

exec /bin/alertmanager --config.file=/tmp/alertmanager.yml
