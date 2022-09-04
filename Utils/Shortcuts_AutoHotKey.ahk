#InstallKeybdHook

^!q::
  Send ^c
  Send ^t
  Send tr
  Send {Enter}
  Sleep, 2000
  Send ^v
Return

^!a::
  Send ^c
  Send #{5}
  Sleep, 500
  Send tr
  Send {Enter}
  Sleep, 2000
  Send ^v
Return

F10::
  Send #p
Return

F4::
  Send !{F4}
Return

f2::
  SoundSet, -5
Return

f3::
  SoundSet, +5
Return

#<::
  Send ^#{right}
Return

#>::
  Send ^#{left}
Return

^q::
  Send ^c
  Send ^t
  Send ^v
  Send {Enter}
Return

^b::
  Send {End}
  Send `;
  Send {Enter}
Return