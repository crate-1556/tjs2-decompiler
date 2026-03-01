# TJS2 Decompiler / TJS2反编译器

This project implements a TJS2 (TJS2100) bytecode decompiler for the Kirikiri visual novel engine, converting compiled bytecode into human-readable and executable TJS2 source code.

用于 Kirikiri（吉里吉里）视觉小说引擎的 TJS2（TJS2100）字节码反编译器，将字节码还原为可读且可执行的 TJS2 源代码。

## Usage / 使用方法

```bash
# Single file / 单文件反编译
python3 tjs2_decompiler.py input.tjs -o output.tjs

# Directory (flat) / 反编译整个文件夹
python3 tjs2_decompiler.py input_dir/ -o output_dir/

# Directory (recursive) / 递归反编译（保持子目录结构）
python3 tjs2_decompiler.py input_dir/ -r -o output_dir/

# Directory (recursive flat) / 递归反编译（输出到同一目录，不保留子目录结构）
python3 tjs2_decompiler.py input_dir/ -f -o output_dir/

# Specify output encoding / 指定输出编码（默认 utf-16le-bom）
python3 tjs2_decompiler.py input.tjs -o output.tjs -e utf-16le-bom
# Supported: utf-8, utf-8-bom, utf-16le-bom, shift_jis, gbk

# Disassemble / 反汇编
python3 tjs2_decompiler.py input.tjs -d

# File info / 查看文件信息
python3 tjs2_decompiler.py input.tjs -i
```

## Example / 示例

Source: [kag3/system/YesNoDialog.tjs](https://github.com/krkrz/kag3/blob/master/data/system/YesNoDialog.tjs)

<details>
<summary> Source Code / 源码</summary>

```javascript
// YesNoDialog.tjs - はい/いいえを選択するダイアログボックス
// Copyright (C)2001-2009, W.Dee and contributors  改変・配布は自由です

class YesNoDialogWindow extends Window
{
	var yesButton; // [はい] ボタン
	var noButton; // [いいえ] ボタン

	var result = false; // no:false yes:true

	function YesNoDialogWindow(message, cap)
	{
		super.Window();

		// メインウィンドウから cursor**** の情報をとってくる
		if(global.Window.mainWindow !== null &&
			typeof global.Window.mainWindow.cursorDefault != "undefined")
			this.cursorDefault = global.Window.mainWindow.cursorDefault;
		if(global.Window.mainWindow !== null &&
			typeof global.Window.mainWindow.cursorPointed != "undefined")
			this.cursorPointed = global.Window.mainWindow.cursorPointed;

		// 外見の調整
		borderStyle = bsDialog;
		innerSunken = false;
		caption = cap;
		showScrollBars = false;

		// プライマリレイヤの作成
		add(new Layer(this, null));

		// プライマリのマウスカーソルを設定
		if(typeof this.cursorDefault !== "undefined")
			primaryLayer.cursor = cursorDefault;

		// サイズを決定
		var tw = primaryLayer.font.getTextWidth(message);
		var th = primaryLayer.font.getTextHeight(message);

		var w = tw + 40;
		if(w<200) w = 200;
		var h = th*2 + 60;

		// 拡大率の設定
		if (kag.fullScreen) {
			if (kag.innerWidth / kag.scWidth
				< kag.innerHeight / kag.scHeight)
				setZoom(kag.innerWidth, kag.scWidth);
			else
				setZoom(kag.innerHeight, kag.scHeight);
		} else {
		  setZoom(kag.zoomNumer, kag.zoomDenom);
		}
		// サイズを決定
		setInnerSize(w * zoomNumer / zoomDenom,
			 h * zoomNumer / zoomDenom);

		// プライマリレイヤのサイズを設定
		primaryLayer.width = w;
		primaryLayer.height = h;
		primaryLayer.colorRect(0, 0, w, h, clBtnFace, 255);

		// ウィンドウ位置の調整
		if(global.Window.mainWindow !== null && global.Window.mainWindow isvalid)
		{
			var win = global.Window.mainWindow;
			var l, t;
			l = ((win.width - width)>>1) + win.left;
			t = ((win.height - height)>>1) + win.top;
			if(l < 0) l = 0;
			if(t < 0) t = 0;
			if(l + width > System.screenWidth) l = System.screenWidth - width;
			if(t + height > System.screenHeight) t = System.screenHeight - height;
			setPos(l, t);
		}
		else
		{
			setPos((System.screenWidth - width)>>1, (System.screenHeight - height)>>1);
		}

		// メッセージの描画
		primaryLayer.drawText((w - tw)>>1, 14, message, clBtnText);

		// Yesボタン
		add(yesButton = new ButtonLayer(this, primaryLayer));
		yesButton.caption = "はい";
		yesButton.captionColor = clBtnText;
		yesButton.width = 70;
		yesButton.height = 25;
		yesButton.top = th + 35;
		yesButton.left = (w - (70*2 + 10)>>1);
		yesButton.visible = true;

		// Noボタン
		add(noButton = new ButtonLayer(this, primaryLayer));
		noButton.caption = "いいえ";
		noButton.captionColor = clBtnText;
		noButton.width = 70;
		noButton.height = 25;
		noButton.top = th + 35;
		noButton.left = ((w - (70*2 + 10))>>1) + 70 + 10;
		noButton.visible = true;

	}

	function finalize()
	{
		super.finalize(...);
	}

	function action(ev)
	{
		if(ev.type == "onClick")
		{
			if(ev.target == yesButton)
			{
				result = true;
				close();
			}
			else if(ev.target == noButton)
			{
				result = false;
				close();
			}
		}
		else if(ev.type == "onKeyDown" && ev.target === this)
		{
			switch(ev.key)
			{
			case VK_PADLEFT:
				yesButton.focus();
				break;
			case VK_PADRIGHT:
				noButton.focus();
				break;
			case VK_PAD1:
				if(focusedLayer == yesButton)
				{
					result = true;
					close();
				}
				else if(focusedLayer == noButton)
				{
					result = false;
					close();
				}
				break;
			case VK_PAD2:
				result = false;
				close();
				break;
			}
		}
	}

	function onKeyDown(key, shift)
	{
		super.onKeyDown(...);
		if(key == VK_ESCAPE)
		{
			result = false;
			close();
		}
	}
}

function askYesNo(message, caption = "確認")
{
	var win = new YesNoDialogWindow(message, caption);
	win.showModal();
	var res = win.result;
	invalidate win;
	return res;
}
```

</details>

<details>
<summary>Decompiled Output / 反编译输出</summary>

```javascript
class YesNoDialogWindow extends Window {
    var yesButton;
    var noButton;
    var result = 0;

    function YesNoDialogWindow(arg0, arg1) {
        super.Window();
        if (super.mainWindow !== null && typeof super.mainWindow.cursorDefault != "undefined") {
            this.cursorDefault = super.mainWindow.cursorDefault;
        }
        if (super.mainWindow !== null && typeof super.mainWindow.cursorPointed != "undefined") {
            this.cursorPointed = super.mainWindow.cursorPointed;
        }
        borderStyle = bsDialog;
        innerSunken = 0;
        caption = arg1;
        showScrollBars = 0;
        add(new Layer(this, null));
        if (typeof this.cursorDefault !== "undefined") {
            primaryLayer.cursor = cursorDefault;
        }
        var local0 = primaryLayer.font.getTextWidth(arg0);
        var local1 = primaryLayer.font.getTextHeight(arg0);
        var local2 = local0 + 40;
        if (local2 < 200) {
            local2 = 200;
        }
        var local3 = local1 * 2 + 60;
        if (kag.fullScreen) {
            if (kag.innerWidth / kag.scWidth < kag.innerHeight / kag.scHeight) {
                setZoom(kag.innerWidth, kag.scWidth);
            } else {
                setZoom(kag.innerHeight, kag.scHeight);
            }
        } else {
            setZoom(kag.zoomNumer, kag.zoomDenom);
        }
        setInnerSize(local2 * zoomNumer / zoomDenom, local3 * zoomNumer / zoomDenom);
        primaryLayer.width = local2;
        primaryLayer.height = local3;
        primaryLayer.colorRect(0, 0, local2, local3, clBtnFace, 255);
        if (super.mainWindow !== null && isvalid super.mainWindow) {
            var local4 = super.mainWindow;
            var local5;
            var local6;
            local5 = (local4.width - width >> 1) + local4.left;
            local6 = (local4.height - height >> 1) + local4.top;
            if (local5 < 0) {
                local5 = 0;
            }
            if (local6 < 0) {
                local6 = 0;
            }
            if (local5 + width > System.screenWidth) {
                local5 = System.screenWidth - width;
            }
            if (local6 + height > System.screenHeight) {
                local6 = System.screenHeight - height;
            }
            setPos(local5, local6);
        } else {
            setPos(System.screenWidth - width >> 1, System.screenHeight - height >> 1);
        }
        primaryLayer.drawText(local2 - local0 >> 1, 14, arg0, clBtnText);
        add((yesButton = new ButtonLayer(this, primaryLayer)));
        yesButton.caption = "はい";
        yesButton.captionColor = clBtnText;
        yesButton.width = 70;
        yesButton.height = 25;
        yesButton.top = local1 + 35;
        yesButton.left = local2 - 150 >> 1;
        yesButton.visible = 1;
        add((noButton = new ButtonLayer(this, primaryLayer)));
        noButton.caption = "いいえ";
        noButton.captionColor = clBtnText;
        noButton.width = 70;
        noButton.height = 25;
        noButton.top = local1 + 35;
        noButton.left = (local2 - 150 >> 1) + 70 + 10;
        noButton.visible = 1;
    }

    function finalize() {
        super.finalize(...);
    }

    function action(arg0) {
        if (arg0.type == "onClick") {
            if (arg0.target == yesButton) {
                result = 1;
                close();
            } else if (arg0.target == noButton) {
                result = 0;
                close();
            }
        } else if (arg0.type == "onKeyDown" && arg0.target === this) {
            switch (arg0.key) {
                case VK_PADLEFT:
                    yesButton.focus();
                    break;
                case VK_PADRIGHT:
                    noButton.focus();
                    break;
                case VK_PAD1:
                    if (focusedLayer == yesButton) {
                        result = 1;
                        close();
                    } else if (focusedLayer == noButton) {
                        result = 0;
                        close();
                    }
                    break;
                case VK_PAD2:
                    result = 0;
                    close();
                    break;
            }
        }
    }

    function onKeyDown(arg0, arg1) {
        super.onKeyDown(...);
        if (arg0 == VK_ESCAPE) {
            result = 0;
            close();
        }
    }
}
this.YesNoDialogWindow = YesNoDialogWindow;

function askYesNo(arg0, arg1 = "確認") {
    var local0 = new YesNoDialogWindow(arg0, arg1);
    local0.showModal();
    var local1 = local0.result;
    invalidate(local0);
    return local1;
}
this.askYesNo = askYesNo incontextof this;
```

</details>

## Validation / 测试

This decompiler has been verified to cover all bytecode patterns the compiler can generate, and validated against all TJS2 scripts in the following directories:  
已验证涵盖所有TJS2编译器可生成的字节码模式，并已对以下目录中的所有TJS2脚本完成验证测试：

- [x] [kag3/system](https://github.com/krkrz/kag3/tree/master/data/system)
- [x] [Krkr2Compat](https://github.com/krkrz/Krkr2Compat)
- [x] [KAGEX3/system](https://github.com/krkrz/krkr2/tree/master/kirikiri2/branches/kag3ex3/template/system) 
- [x] All script files of a complete game / 一部游戏的全部脚本资源

The complete game has been runtime-verified to launch and run correctly using only decompiled scripts.  
游戏测试项目已通过运行时验证，仅使用反编译脚本即可正常启动并运行。