# TJS2 Decompiler / TJS2反编译器

This project implements a TJS2 (TJS2100) bytecode decompiler for the Kirikiri visual novel engine, converting compiled bytecode into human-readable TJS2 source code.

用于 Kirikiri（吉里吉里）视觉小说引擎的 TJS2（TJS2100）字节码反编译器，将字节码还原为可读的 TJS2 源代码。

## Usage / 使用方法

```bash
# Single file / 单文件反编译
python3 tjs2_decompiler.py input.tjs -o output.tjs

# Directory (flat) / 反编译整个文件夹
python3 tjs2_decompiler.py input_dir/ -o output_dir/

# Directory (recursive) / 递归反编译（保持子目录结构）
python3 tjs2_decompiler.py input_dir/ -r -o output_dir/

# Disassemble / 反汇编
python3 tjs2_decompiler.py input.tjs -d

# File info / 查看文件信息
python3 tjs2_decompiler.py input.tjs -i
```

## Example / 示例

Source: [kag3/YesNoDialog.tjs](https://github.com/krkrz/kag3/blob/master/data/system/YesNoDialog.tjs)

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
this.YesNoDialogWindow = YesNoDialogWindow;
this.askYesNo = askYesNo incontextof this;

class YesNoDialogWindow {
    (this.Window incontextof this)();
    this.yesButton = void;
    this.noButton = void;
    this.result = 0;

    function YesNoDialogWindow(arg0, arg1) {
        global.Window.Window();
        if (global.Window.mainWindow !== null && typeof global.Window.mainWindow.cursorDefault != "undefined") {
            this.cursorDefault = global.Window.mainWindow.cursorDefault;
        }
        if (global.Window.mainWindow !== null && typeof global.Window.mainWindow.cursorPointed != "undefined") {
            this.cursorPointed = global.Window.mainWindow.cursorPointed;
        }
        this.borderStyle = this.bsDialog;
        this.innerSunken = 0;
        this.caption = arg1;
        this.showScrollBars = 0;
        this.add(new this.Layer(this, null));
        if (typeof this.cursorDefault !== "undefined") {
            this.primaryLayer.cursor = this.cursorDefault;
        }
        var local0 = this.primaryLayer.font.getTextWidth(arg0);
        var local1 = this.primaryLayer.font.getTextHeight(arg0);
        var local2 = local0 + 40;
        if (local2 < 200) {
            local2 = 200;
        }
        var local3 = local1 * 2 + 60;
        if (this.kag.fullScreen) {
            if (this.kag.innerWidth / this.kag.scWidth < this.kag.innerHeight / this.kag.scHeight) {
                this.setZoom(this.kag.innerWidth, this.kag.scWidth);
            } else {
                this.setZoom(this.kag.innerHeight, this.kag.scHeight);
            }
        } else {
            this.setZoom(this.kag.zoomNumer, this.kag.zoomDenom);
        }
        this.setInnerSize(local2 * this.zoomNumer / this.zoomDenom, local3 * this.zoomNumer / this.zoomDenom);
        this.primaryLayer.width = local2;
        this.primaryLayer.height = local3;
        this.primaryLayer.colorRect(0, 0, local2, local3, this.clBtnFace, 255);
        if (global.Window.mainWindow !== null && isvalid global.Window.mainWindow) {
            var local4 = global.Window.mainWindow;
            var local5;
            var local6;
            local5 = (local4.width - this.width >> 1) + local4.left;
            local6 = (local4.height - this.height >> 1) + local4.top;
            if (local5 < 0) {
                local5 = 0;
            }
            if (local6 < 0) {
                local6 = 0;
            }
            if (local5 + this.width > this.System.screenWidth) {
                local5 = this.System.screenWidth - this.width;
            }
            if (local6 + this.height > this.System.screenHeight) {
                local6 = this.System.screenHeight - this.height;
            }
            this.setPos(local5, local6);
        } else {
            this.setPos(this.System.screenWidth - this.width >> 1, this.System.screenHeight - this.height >> 1);
        }
        this.primaryLayer.drawText(local2 - local0 >> 1, 14, arg0, this.clBtnText);
        this.add(this.yesButton = new this.ButtonLayer(this, this.primaryLayer));
        this.yesButton.caption = "はい";
        this.yesButton.captionColor = this.clBtnText;
        this.yesButton.width = 70;
        this.yesButton.height = 25;
        this.yesButton.top = local1 + 35;
        this.yesButton.left = local2 - 150 >> 1;
        this.yesButton.visible = 1;
        this.add(this.noButton = new this.ButtonLayer(this, this.primaryLayer));
        this.noButton.caption = "いいえ";
        this.noButton.captionColor = this.clBtnText;
        this.noButton.width = 70;
        this.noButton.height = 25;
        this.noButton.top = local1 + 35;
        this.noButton.left = (local2 - 150 >> 1) + 70 + 10;
        this.noButton.visible = 1;
    }

    function finalize() {
        global.Window.finalize(...);
    }

    function action(arg0) {
        if (arg0.type == "onClick") {
            if (arg0.target == this.yesButton) {
                this.result = 1;
                this.close();
            } else if (arg0.target == this.noButton) {
                this.result = 0;
                this.close();
            }
        } else if (arg0.type == "onKeyDown" && arg0.target === this) {
            switch (arg0.key) {
            case this.VK_PADLEFT:
                this.yesButton.focus();
                break;
            case this.VK_PADRIGHT:
                this.noButton.focus();
                break;
            case this.VK_PAD1:
                if (this.focusedLayer == this.yesButton) {
                    this.result = 1;
                    this.close();
                } else if (this.focusedLayer == this.noButton) {
                    this.result = 0;
                    this.close();
                }
                break;
            case this.VK_PAD2:
                this.result = 0;
                this.close();
                break;
            }
        }
    }

    function onKeyDown(arg0, arg1) {
        global.Window.onKeyDown(...);
        if (arg0 == this.VK_ESCAPE) {
            this.result = 0;
            this.close();
        }
    }
}

function askYesNo(arg0, arg1) {
    if (arg1 === void) {
        arg1 = "確認";
    }
    var local0 = new this.YesNoDialogWindow(arg0, arg1);
    local0.showModal();
    var local1 = local0.result;
    invalidate(local0);
    return local1;
}
```

</details>

## Validated Against / 已验证测试目录

The decompiler has been validated against all TJS2 scripts contained within the following directories:

已对以下目录中的所有TJS2脚本进行验证测试：

- [x] [kag3/system](https://github.com/krkrz/kag3/tree/master/data/system)
- [x] [Krkr2Compat](https://github.com/krkrz/Krkr2Compat)
- [x] [KAGEX3/system](https://github.com/krkrz/krkr2/tree/master/kirikiri2/branches/kag3ex3/template/system) 
- [ ] All script files of a complete game / 一部游戏的全部脚本资源 （PLAN）

**NOTE**: 前3个dir还存在少量BUG，以函数为单位目前准确率在99%左右，等待近期下一次更新。